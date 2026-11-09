# Joint LS-ViT: whole-network joint optimisation of the AdaRound rounding
# variables on top of the original LS-ViT reconstruction math.
#
# This module only adds an optimisation structure.  Module discovery, AdaRound
# wrapping, the least-squares sensitivity formulas and the final hard-rounding
# conversion are reused from the original implementation; the per-block entry
# point (BlockReconstructor.reconstruct_single_block) is never called.
#
# Loss implemented here (all terms computed on the same image batch):
#
#   L_joint = L_LS_joint + lambda_KL * L_KL_global + L_round
#
#   L_LS_joint = mean_{b in recon modules} ( L_D,b / c_D,b + L_U,b / c_U,b )
#   L_D,b      = mean_n ( sum_k d[b,k] * delta[b,k]^2 )
#   L_U,b      = mean_n ( ( sum_k u[b,k] * |delta[b,k]| )^2 )
#   delta[b]   = quantised_full_network[b] - fp32_reference[b]
#   L_KL       = KL( log_softmax(quant_logits / T) || softmax(fp_logits / T) )
#   L_round    = mean_{b all modules} sum_j 0.01 * (1 - |2h(alpha) - 1|^beta(t))
#
# d[b,k] and u[b,k] are the LS-ViT least-squares sensitivities
#
#   v[n] = sum_k a[n,k] * r[n,k]
#   d[k] = mean_n(a[n,k] * r[n,k]) / mean_n(r[n,k]^2)
#   u[k] = mean_n(a[n,k] * sqrt(v[n])) / mean_n(v[n])
#
# with a = |d KL / d module output| and r = |quantised - fp32| of the same
# module, collected from a single-module quantised perturbation while every
# other module runs in FP32.
import math

import torch
import torch.nn.functional as F

from quant_layers import MinMaxQuantLinear, MinMaxQuantConv2d, MinMaxQuantMatMul
from utils.block_recon import BlockReconstructor, LinearTempDecay


class JointLSViTReconstructor:
    """Joint optimisation of every LS-ViT reconstruction module in one graph."""

    def __init__(self, model, calib_loader, batch_size=32, temp=20,
                 iters=20000, weight=0.01, b_range=(20, 2), warmup=0.2,
                 lr=4e-5, drop_prob=1.0, lambda_kl=None):
        self.model = model
        self.calib_loader = calib_loader
        self.batch_size = batch_size
        self.temp = temp
        self.iters = iters
        self.weight = weight
        self.b_range = b_range
        self.warmup = warmup
        self.lr = lr
        self.drop_prob = drop_prob
        self.lambda_kl = lambda_kl

        # Reused only for module discovery and the original AdaRound wrapping
        # rules; no per-block training is ever invoked.
        self.br = BlockReconstructor(model, batch_size, calib_loader,
                                     metric='lsvit', temp=temp)
        self.ls_modules = [name for name in self.br.blocks if name != 'head']
        self.device = next(model.parameters()).device
        self.dataset = calib_loader.dataset
        self.loss_start = warmup * iters
        self.temp_decay = LinearTempDecay(iters, rel_start_decay=warmup,
                                          start_b=b_range[0], end_b=b_range[1])

        self.d = {}
        self.u = {}
        self.c_D = {}
        self.c_U = {}
        self.w_params = []
        self.a_params = []
        self.w_opt = None
        self.a_opt = None
        self.a_sched = None

    # ------------------------------------------------------------------ setup

    def prepare(self):
        """Initialise all quantizers, fit LS sensitivities, fix c_D/c_U."""
        self.init_all_quantizers()
        self.collect_sensitivity()
        self.compute_norm_constants()

    def init_all_quantizers(self):
        """Wrap every weight quantizer in AdaRound and build both optimizers.

        All modules are initialised in one pass, before any optimiser update, so
        no block can be trained while another one is still in FP32 state.
        """
        for name, module in self.br.blocks.items():
            self.br.wrap_quantizers_in_net(module, name)

        for name, module in self.br.blocks.items():
            for _, m in module.named_modules():
                if hasattr(m, 'training_mode'):
                    m.init_training()
            self.br.set_qdrop(module, self.drop_prob)

        # Freeze every parameter once all AdaRound variables exist; only the
        # rounding variables and the activation scales are re-opened in
        # _build_optimizers.  Freezing does not cut the graph between blocks:
        # the forward path still connects every module output.
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        self._set_all_mode('quant_forward')
        self._build_optimizers()

    def _build_optimizers(self):
        w_params, a_params = [], []
        for name, module in self.br.blocks.items():
            for _, m in module.named_modules():
                if isinstance(m, (MinMaxQuantLinear, MinMaxQuantConv2d)):
                    m.w_quantizer.alpha.requires_grad = True
                    m.a_quantizer.scale.requires_grad = True
                    w_params.append(m.w_quantizer.alpha)
                    a_params.append(m.a_quantizer.scale)
                elif isinstance(m, MinMaxQuantMatMul):
                    m.A_quantizer.scale.requires_grad = True
                    m.B_quantizer.scale.requires_grad = True
                    a_params.append(m.A_quantizer.scale)
                    a_params.append(m.B_quantizer.scale)

        self.w_params = w_params
        self.a_params = a_params
        self.w_opt = torch.optim.Adam(w_params)
        self.a_opt = torch.optim.Adam(a_params, lr=self.lr)
        self.a_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.a_opt, T_max=self.iters, eta_min=0.)

    # ----------------------------------------------------------- sensitivities

    def collect_sensitivity(self):
        """Fit d_b and u_b with the original LS-ViT least-squares formulas.

        State during collection: every rounding variable is initialised, no
        joint update has run, upstream modules are FP32, the current module runs
        its initialised quantised forward, downstream modules are FP32 and the
        teacher is the FP32 softmax of the same batch.  No parameter is updated.
        """
        self._set_all_mode('raw')
        acc = {}
        n_images = 0
        for inp, _ in self.calib_loader:
            inp = inp.to(self.device)
            b = inp.shape[0]
            n_images += b
            # _perturbed_backward leaves the module it last perturbed in
            # quant_forward mode, so the FP32 state has to be restored at the
            # start of every batch; doing it once before the loop is not enough.
            self._set_all_mode('raw')
            with torch.no_grad():
                raw_outs, fp_logits = self._capture(inp)
            teacher = F.softmax(fp_logits / self.temp, dim=-1).detach()

            for name in self.ls_modules:
                q_out, grad_out = self._perturbed_backward(inp, name, teacher)
                r = (q_out - raw_outs[name]).reshape(b, -1).abs()
                a = grad_out.reshape(b, -1).abs()
                v = (a * r).sum(-1)
                if name not in acc:
                    acc[name] = {
                        'ar': torch.zeros(r.shape[1], device=self.device),
                        'r2': torch.zeros(r.shape[1], device=self.device),
                        'asv': torch.zeros(r.shape[1], device=self.device),
                        'v': torch.zeros((), device=self.device),
                        'r_max': 0.0,
                        'r_zero': 0,
                        'r_numel': 0,
                    }
                acc[name]['ar'] += (a * r).sum(dim=0)
                acc[name]['r2'] += (r * r).sum(dim=0)
                acc[name]['asv'] += (a * v.sqrt().unsqueeze(1)).sum(dim=0)
                acc[name]['v'] += v.sum()
                acc[name]['r_max'] = max(acc[name]['r_max'], float(r.max()))
                acc[name]['r_zero'] += int((r == 0).sum())
                acc[name]['r_numel'] += r.numel()

        for name in self.ls_modules:
            stats = acc[name]
            bad_r2 = int((stats['r2'] <= 0).sum())
            if bad_r2 > 0 or not bool(torch.isfinite(stats['r2']).all()):
                raise RuntimeError(
                    'degenerate sensitivity for {}: mean(r^2) denominator has '
                    '{} non-positive of {} entries, min={!r}, max={!r}, '
                    'all_finite={}'.format(name, bad_r2, stats['r2'].numel(),
                                           float(stats['r2'].min()),
                                           float(stats['r2'].max()),
                                           bool(torch.isfinite(stats['r2']).all())))
            if not bool(torch.isfinite(stats['v'])) or float(stats['v']) <= 0.0:
                raise RuntimeError(
                    'degenerate sensitivity for {}: sum_n(v)={!r}'.format(
                        name, float(stats['v'])))

            d = stats['ar'] / stats['r2']
            u = stats['asv'] / stats['v']
            if not bool(torch.isfinite(d).all()) or not bool(torch.isfinite(u).all()):
                raise RuntimeError('non-finite sensitivity d/u for {}'.format(name))

            self.d[name] = d.reshape(1, -1)
            self.u[name] = u.reshape(1, -1)
            r_rms = math.sqrt(float(stats['r2'].sum()) / stats['r_numel'])
            r_zero = stats['r_zero'] / stats['r_numel']
            print('sensitivity {:14s}: d mean={:.4e} min={:.4e} max={:.4e} | '
                  'u mean={:.4e} min={:.4e} max={:.4e} | '
                  'r rms={:.4e} max={:.4e} zero={:.4%}'.format(
                      name, float(d.mean()), float(d.min()), float(d.max()),
                      float(u.mean()), float(u.min()), float(u.max()),
                      r_rms, stats['r_max'], r_zero))

        print('sensitivity collected: {} modules, {} images'.format(
            len(self.ls_modules), n_images))

    def _perturbed_backward(self, inp, name, teacher):
        """Run one module quantised, everything else FP32, and step back to it."""
        self._set_all_mode('raw')
        self.br.set_block_mode(self.br.blocks[name], 'quant_forward')
        store = {}

        def forward_hook(module, inputs, output):
            store['out'] = output

        def backward_hook(module, grad_input, grad_output):
            store['grad'] = grad_output[0]

        handle_f = self.br.blocks[name].register_forward_hook(forward_hook)
        handle_b = self.br.blocks[name].register_full_backward_hook(backward_hook)
        try:
            self._zero_grads()
            logits = self.model(inp)
            loss = F.kl_div(F.log_softmax(logits / self.temp, dim=-1),
                            teacher, reduction='batchmean')
            loss.backward()
        finally:
            handle_f.remove()
            handle_b.remove()

        q_out = store['out'].detach()
        grad_out = store['grad'].detach()
        self._zero_grads()
        return q_out, grad_out

    # ----------------------------------------------------------- normalisers

    def compute_norm_constants(self):
        """Fix c_D,b and c_U,b on the full quantised network, before training.

        Measured once on the whole optimisation calibration set with every
        module quantised from its initialised state; the values stay fixed for
        the whole joint optimisation.
        """
        self._set_all_mode('raw')
        ld = {name: torch.zeros((), device=self.device) for name in self.ls_modules}
        lu = {name: torch.zeros((), device=self.device) for name in self.ls_modules}
        n_images = 0

        for inp, _ in self.calib_loader:
            inp = inp.to(self.device)
            b = inp.shape[0]
            n_images += b
            with torch.no_grad():
                raw_outs, _ = self._capture(inp)
                self._set_all_mode('quant_forward')
                q_outs, _ = self._capture(inp)
                self._set_all_mode('raw')
            for name in self.ls_modules:
                cha = (q_outs[name] - raw_outs[name]).reshape(b, -1).abs()
                ld[name] += (cha.pow(2) * self.d[name]).sum(-1).sum()
                lu[name] += ((cha * self.u[name]).sum(-1).pow(2)).sum()

        for name in self.ls_modules:
            c_d = float(ld[name] / n_images)
            c_u = float(lu[name] / n_images)
            print('normaliser {:14s}: c_D={:.6e} c_U={:.6e}'.format(name, c_d, c_u))
            if (not math.isfinite(c_d) or not math.isfinite(c_u)
                    or c_d <= 0.0 or c_u <= 0.0):
                raise RuntimeError(
                    'degenerate reconstruction normaliser for {}: c_D={!r} '
                    'c_U={!r} (full quantised network, {} images)'.format(
                        name, c_d, c_u, n_images))
            self.c_D[name] = c_d
            self.c_U[name] = c_u

        print('normalisers fixed on {} images'.format(n_images))

    # ----------------------------------------------------------------- losses

    def _ls_loss(self, q_outs, fp_outs):
        total = None
        for name in self.ls_modules:
            cha = (q_outs[name] - fp_outs[name]).reshape(fp_outs[name].shape[0], -1).abs()
            l_d = (cha.pow(2) * self.d[name]).sum(-1).mean()
            l_u = (cha * self.u[name]).sum(-1).pow(2).mean()
            term = l_d / self.c_D[name] + l_u / self.c_U[name]
            total = term if total is None else total + term
        return total / len(self.ls_modules)

    def _kl_loss(self, q_logits, fp_logits):
        teacher = F.softmax(fp_logits / self.temp, dim=-1).detach()
        return F.kl_div(F.log_softmax(q_logits / self.temp, dim=-1),
                        teacher, reduction='batchmean')

    def _round_loss(self, b):
        total = None
        for name, module in self.br.blocks.items():
            for _, m in module.named_modules():
                if hasattr(m, 'w_quantizer'):
                    vals = m.w_quantizer.get_soft_targets()
                    term = self.weight * (1 - ((vals - .5).abs() * 2).pow(b)).sum()
                    total = term if total is None else total + term
        return total / len(self.br.blocks)

    def _ls_term(self, name, q_outs, fp_outs):
        """One module's LS-ViT reconstruction term.

        Mirrors the corresponding addend of _ls_loss exactly; it exists so the
        diagnosis can backpropagate a single module's term on its own.
        """
        cha = (q_outs[name] - fp_outs[name]).reshape(fp_outs[name].shape[0], -1).abs()
        l_d = (cha.pow(2) * self.d[name]).sum(-1).mean()
        l_u = (cha * self.u[name]).sum(-1).pow(2).mean()
        return l_d / self.c_D[name] + l_u / self.c_U[name]

    # --------------------------------------------------------------- training

    def train(self):
        """One graph per step: FP32 teacher, quantised student, single backward."""
        if self.lambda_kl is None:
            raise ValueError('lambda_kl must be given explicitly before joint training')

        n = len(self.dataset)
        for it in range(self.iters):
            idx = torch.randperm(n)[:self.batch_size].tolist()
            inp = torch.stack([self.dataset[i][0] for i in idx]).to(self.device)

            self._set_all_mode('raw')
            with torch.no_grad():
                fp_outs, fp_logits = self._capture(inp)

            self._set_all_mode('quant_forward')
            q_outs, q_logits = self._capture(inp)

            loss_ls = self._ls_loss(q_outs, fp_outs)
            loss_kl = self._kl_loss(q_logits, fp_logits)

            count = it + 1
            if count < self.loss_start:
                b = 0.0
                loss_round = torch.zeros((), device=self.device)
            else:
                b = self.temp_decay(count)
                loss_round = self._round_loss(b)

            loss = loss_ls + self.lambda_kl * loss_kl + loss_round

            self.w_opt.zero_grad()
            self.a_opt.zero_grad()
            loss.backward()
            self.w_opt.step()
            self.a_opt.step()
            self.a_sched.step()

            if count == 1 or count % 500 == 0:
                print('Joint loss:\t{:.4f} (ls:{:.4f}, kl:{:.4e}, '
                      'kl*lambda:{:.4f}, round:{:.4f})\tb={:.2f}\tcount={}'.format(
                          float(loss), float(loss_ls), float(loss_kl),
                          float(self.lambda_kl * loss_kl), float(loss_round),
                          b, count))

    def finalize(self):
        """Switch every module to hard rounding and write the weights back."""
        for _, m in self.model.named_modules():
            if hasattr(m, 'w_quantizer'):
                m.w_quantizer.soft_targets = False
            if hasattr(m, 'training_mode'):
                m.end_training()
            if hasattr(m, 'mode'):
                m.mode = 'quant_forward'

        for _, m in self.model.named_modules():
            if hasattr(m, 'w_quantizer'):
                m.weight.data.copy_(m.w_quantizer.get_hard_value(m.weight.data))
                del m.w_quantizer.alpha
                m.w_quantizer.round_mode = 'nearest'

    # -------------------------------------------------------------- diagnosis

    def diagnose(self, batches=4):
        """Measure the unweighted LS/KL/round terms and their gradient scales.

        No optimiser step is taken and no lambda_KL is derived.  Terms are
        averaged over `batches` batches of the optimisation calibration set.
        Gradient norms are those of the mean gradient over those batches;
        cosines are computed per batch and then averaged over the batches for
        which both gradients are non-zero.
        """
        self._set_all_mode('raw')
        ls_sum = 0.0
        kl_sum = 0.0
        round_sum = 0.0
        ld_sum = {name: 0.0 for name in self.ls_modules}
        lu_sum = {name: 0.0 for name in self.ls_modules}
        grad_ls = {}
        grad_round = {}
        grad_kl = {}
        scale_ls = {}
        scale_kl = {}
        cross_b11 = {}
        cos_alpha = {name: [] for name in self.ls_modules}
        cos_global = []
        used = 0

        for i in range(batches):
            low = i * self.batch_size
            high = low + self.batch_size
            inp = torch.stack([self.dataset[j][0] for j in range(low, high)]).to(self.device)

            self._set_all_mode('raw')
            with torch.no_grad():
                fp_outs, fp_logits = self._capture(inp)
            self._set_all_mode('quant_forward')
            q_outs, q_logits = self._capture(inp)

            loss_ls = self._ls_loss(q_outs, fp_outs)
            loss_kl = self._kl_loss(q_logits, fp_logits)
            loss_round = self._round_loss(20.0)
            loss_b11 = self._ls_term('blocks.11', q_outs, fp_outs)
            ls_sum += float(loss_ls)
            kl_sum += float(loss_kl)
            round_sum += float(loss_round)
            for name in self.ls_modules:
                cha = (q_outs[name] - fp_outs[name]).reshape(inp.shape[0], -1).abs()
                ld_sum[name] += float((cha.pow(2) * self.d[name]).sum(-1).mean())
                lu_sum[name] += float((cha * self.u[name]).sum(-1).pow(2).mean())

            self._zero_grads()
            loss_ls.backward(retain_graph=True)
            cur_ls = self._alpha_grads()
            cur_scale_ls = self._scale_grads()
            self._zero_grads()
            loss_b11.backward(retain_graph=True)
            cur_b11 = self._alpha_grads()
            self._zero_grads()
            loss_round.backward(retain_graph=True)
            cur_round = self._alpha_grads()
            self._zero_grads()
            loss_kl.backward()
            cur_kl = self._alpha_grads()
            cur_scale_kl = self._scale_grads()
            self._zero_grads()

            # Direction is measured on this batch alone, before any averaging.
            for name in self.ls_modules:
                cos_alpha[name].append(self._cosine(cur_ls[name], cur_kl[name]))
            cos_global.append(self._cosine(torch.cat([cur_ls[n] for n in cur_ls]),
                                           torch.cat([cur_kl[n] for n in cur_kl])))

            for name in cur_ls:
                grad_ls[name] = cur_ls[name] if name not in grad_ls else grad_ls[name] + cur_ls[name]
                grad_round[name] = (cur_round[name] if name not in grad_round
                                    else grad_round[name] + cur_round[name])
                grad_kl[name] = cur_kl[name] if name not in grad_kl else grad_kl[name] + cur_kl[name]
            for name in cur_scale_ls:
                scale_ls[name] = (cur_scale_ls[name] if name not in scale_ls
                                  else scale_ls[name] + cur_scale_ls[name])
                scale_kl[name] = (cur_scale_kl[name] if name not in scale_kl
                                  else scale_kl[name] + cur_scale_kl[name])
            for name in ('blocks.0', 'blocks.11'):
                cross_b11[name] = (cur_b11[name] if name not in cross_b11
                                   else cross_b11[name] + cur_b11[name])
            used += 1

        print('=== joint scale diagnosis: {} batches x {} images ==='.format(
            used, self.batch_size))
        print('L_LS (unweighted, mean over {} LS modules) = {:.6e}'.format(
            len(self.ls_modules), ls_sum / used))
        print('L_KL (unweighted, T={}) = {:.6e}'.format(self.temp, kl_sum / used))
        print('L_R (AdaRound, beta=20, mean over {} modules) = {:.6e}'.format(
            len(self.br.blocks), round_sum / used))
        print('{:14s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}'.format(
            'module', 'L_D/c_D', 'L_U/c_U', '||g_LS||', '||g_R||', '||g_KL||'))
        for name in self.br.blocks.keys():
            if name in ld_sum:
                cell_d = '{:.4e}'.format(ld_sum[name] / used / self.c_D[name])
                cell_u = '{:.4e}'.format(lu_sum[name] / used / self.c_U[name])
            else:
                cell_d = '-'
                cell_u = '-'
            norm_ls = float(grad_ls[name].norm() / used) if name in grad_ls else 0.0
            norm_r = float(grad_round[name].norm() / used) if name in grad_round else 0.0
            norm_kl = float(grad_kl[name].norm() / used) if name in grad_kl else 0.0
            print('{:14s} {:>12s} {:>12s} {:>12.4e} {:>12.4e} {:>12.4e}'.format(
                name, cell_d, cell_u, norm_ls, norm_r, norm_kl))

        all_ls = torch.cat([grad_ls[name] / used for name in grad_ls])
        all_round = torch.cat([grad_round[name] / used for name in grad_round])
        all_kl = torch.cat([grad_kl[name] / used for name in grad_kl])
        print('global ||g_alpha_LS|| = {:.6e}'.format(float(all_ls.norm())))
        print('global ||g_alpha_R||  = {:.6e}'.format(float(all_round.norm())))
        print('global ||g_alpha_KL|| = {:.6e}'.format(float(all_kl.norm())))

        print('--- activation-scale gradients (norm of mean gradient over {} '
              'batches) ---'.format(used))
        print('{:14s} {:>16s} {:>16s}'.format('module', '||g_scale_LS||', '||g_scale_KL||'))
        for name in self.br.blocks.keys():
            print('{:14s} {:>16.6e} {:>16.6e}'.format(
                name, float(scale_ls[name].norm() / used),
                float(scale_kl[name].norm() / used)))
        all_scale_ls = torch.cat([scale_ls[name] / used for name in scale_ls])
        all_scale_kl = torch.cat([scale_kl[name] / used for name in scale_kl])
        print('{:14s} {:>16.6e} {:>16.6e}'.format(
            'global', float(all_scale_ls.norm()), float(all_scale_kl.norm())))

        print('--- alpha gradient direction (cosine per batch, then mean over '
              'defined batches) ---')
        print('{:14s} {:>12s} {:>12s} {:>12s} {:>10s}'.format(
            'module', 'mean cos', 'min', 'max', 'defined'))
        rows = list(self.ls_modules) + ['global']
        for name in rows:
            values = cos_global if name == 'global' else cos_alpha[name]
            defined = [v for v in values if v is not None]
            if defined:
                print('{:14s} {:>12.6f} {:>12.6f} {:>12.6f} {:>7d}/{:<2d}'.format(
                    name, sum(defined) / len(defined), min(defined),
                    max(defined), len(defined), used))
            else:
                print('{:14s} {:>12s} {:>12s} {:>12s} {:>7d}/{:<2d}'.format(
                    name, 'undefined', '-', '-', 0, used))

        print('--- cross-block propagation of L_LS,blocks.11 ---')
        n_up = float(cross_b11['blocks.0'].norm() / used)
        n_own = float(cross_b11['blocks.11'].norm() / used)
        print('||grad_alpha_blocks.0  L_LS,blocks.11|| = {:.6e}'.format(n_up))
        print('||grad_alpha_blocks.11 L_LS,blocks.11|| = {:.6e}'.format(n_own))
        if n_own > 0.0:
            print('upstream / own-block ratio = {:.6e}'.format(n_up / n_own))
        else:
            print('upstream / own-block ratio = undefined (own-block gradient is zero)')

    def _alpha_grads(self):
        grads = {}
        for name, module in self.br.blocks.items():
            chunks = []
            for _, m in module.named_modules():
                if hasattr(m, 'w_quantizer') and m.w_quantizer.alpha is not None:
                    grad = m.w_quantizer.alpha.grad
                    if grad is None:
                        grad = torch.zeros_like(m.w_quantizer.alpha)
                    chunks.append(grad.reshape(-1).detach().clone())
            grads[name] = torch.cat(chunks)
        return grads

    def _scale_grads(self):
        """Concatenated activation-scale gradients, one vector per module.

        Covers exactly the parameters _build_optimizers registers in a_params:
        a_quantizer.scale of Linear/Conv and A/B_quantizer.scale of MatMul.
        Frozen FP32 weights and AdaRound alphas are not included.
        """
        grads = {}
        for name, module in self.br.blocks.items():
            chunks = []
            for _, m in module.named_modules():
                if isinstance(m, (MinMaxQuantLinear, MinMaxQuantConv2d)):
                    scales = [m.a_quantizer.scale]
                elif isinstance(m, MinMaxQuantMatMul):
                    scales = [m.A_quantizer.scale, m.B_quantizer.scale]
                else:
                    continue
                for param in scales:
                    grad = param.grad
                    if grad is None:
                        grad = torch.zeros_like(param)
                    chunks.append(grad.reshape(-1).detach().clone())
            grads[name] = torch.cat(chunks)
        return grads

    @staticmethod
    def _cosine(a, b):
        """Cosine similarity, or None when either vector is all zeros."""
        norm_a = float(a.norm())
        norm_b = float(b.norm())
        if norm_a == 0.0 or norm_b == 0.0:
            return None
        return float(torch.dot(a, b) / (norm_a * norm_b))

    # ----------------------------------------------------------------- helpers

    def _set_all_mode(self, mode):
        for _, m in self.model.named_modules():
            if hasattr(m, 'mode'):
                m.mode = mode

    def _zero_grads(self):
        self.model.zero_grad(set_to_none=True)

    def _capture(self, inp):
        """Forward the model and keep the outputs of the LS modules."""
        outs = {}
        handles = []
        for name in self.ls_modules:
            def hook(module, inputs, output, name=name):
                outs[name] = output
            handles.append(self.br.blocks[name].register_forward_hook(hook))
        try:
            logits = self.model(inp)
        finally:
            for handle in handles:
                handle.remove()
        return outs, logits
