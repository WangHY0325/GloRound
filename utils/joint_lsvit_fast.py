# Joint LS-ViT: cached-input / cached-teacher variant of the joint optimiser.
#
# This module adds no new optimisation maths.  It subclasses
# JointLSViTReconstructor and replaces exactly two things inside the training
# loop:
#
#   1. the optimisation images are materialised once into a contiguous tensor
#      instead of being re-stacked from the CacheDataset on every step;
#   2. the FP32 teacher outputs (the 13 LS module outputs plus the logits) are
#      computed once for all 1024 images and reused.
#
# Everything else -- AdaRound initialisation, the LS sensitivity fit, the fixed
# normalisers, both losses, the AdaRound regulariser, the optimisers, the
# scheduler and finalize() -- is inherited unchanged, so the optimisation
# target and the update sequence are identical to the parent implementation.
import torch

from utils.joint_lsvit import JointLSViTReconstructor


class JointLSViTReconstructorFast(JointLSViTReconstructor):
    """Joint optimisation fed from cached inputs and a cached FP32 teacher."""

    def __init__(self, model, calib_loader, cache_device='cpu', **kwargs):
        super().__init__(model, calib_loader, **kwargs)
        self.cache_device = cache_device
        self.images = None
        self.teacher_outs = None
        self.teacher_logits = None

    def build_caches(self):
        """Materialise the images and the FP32 teacher.

        Called once after prepare(): at that point the quantisation structure
        and the frozen FP32 weights are final, so the teacher is a pure
        function of the input image.
        """
        self.images = torch.stack(
            [self.dataset[i][0] for i in range(len(self.dataset))], dim=0)

        outs = {name: [] for name in self.ls_modules}
        logits = []
        self._set_all_mode('raw')
        with torch.no_grad():
            for inp, _ in self.calib_loader:
                inp = inp.to(self.device)
                batch_outs, batch_logits = self._capture(inp)
                for name in self.ls_modules:
                    outs[name].append(batch_outs[name].to(self.cache_device))
                logits.append(batch_logits.to(self.cache_device))
        self._set_all_mode('quant_forward')

        self.teacher_outs = {name: torch.cat(outs[name], dim=0) for name in self.ls_modules}
        self.teacher_logits = torch.cat(logits, dim=0)

        image_bytes = self.images.numel() * self.images.element_size()
        teacher_floats = (sum(t.numel() for t in self.teacher_outs.values())
                          + self.teacher_logits.numel())
        print('cache built: {} images ({:.0f} MiB), {} FP32 teacher floats '
              '({:.0f} MiB), teacher device={}'.format(
                  self.images.shape[0], image_bytes / 2 ** 20, teacher_floats,
                  teacher_floats * 4 / 2 ** 20, self.cache_device))

    def train(self):
        """Same optimisation as the parent, fed from the caches."""
        if self.lambda_kl is None:
            raise ValueError('lambda_kl must be given explicitly before joint training')

        n = len(self.dataset)
        # The FP32 teacher is no longer evaluated inside the loop, so no step
        # switches the network to raw mode any more.
        self._set_all_mode('quant_forward')
        for it in range(self.iters):
            idx = torch.randperm(n)[:self.batch_size].tolist()
            inp = self.images[idx].to(self.device)
            fp_outs = {name: self.teacher_outs[name][idx].to(self.device)
                       for name in self.ls_modules}
            fp_logits = self.teacher_logits[idx].to(self.device)

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
