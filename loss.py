import torch


class ConRegGroupLoss(torch.autograd.Function):
    def __init__(self):
        self.ad_mmse_frequency_dict = dict()
        self.ad_mmse_frequency_dict_completion = dict()
        self.cn_mmse_frequency_dict = dict()
        self.cn_mmse_frequency_dict_completion = dict()

    def update(self, label_list, mmse_list):
        self.ad_mmse_frequency_dict.clear()
        self.ad_mmse_frequency_dict_completion.clear()
        self.cn_mmse_frequency_dict.clear()
        self.cn_mmse_frequency_dict_completion.clear()
        for i in range(len(label_list)):
            if label_list[i] == 0:
                if mmse_list[i] not in self.cn_mmse_frequency_dict:
                    self.cn_mmse_frequency_dict[mmse_list[i]] = 1
                else:
                    self.cn_mmse_frequency_dict[mmse_list[i]] += 1
            else:
                if mmse_list[i] not in self.ad_mmse_frequency_dict:
                    self.ad_mmse_frequency_dict[mmse_list[i]] = 1
                else:
                    self.ad_mmse_frequency_dict[mmse_list[i]] += 1
        self.cn_mmse_frequency_dict_completion = self.completion(self.cn_mmse_frequency_dict)
        self.ad_mmse_frequency_dict_completion = self.completion(self.ad_mmse_frequency_dict)

        cn_frequency = torch.zeros((2, len(self.cn_mmse_frequency_dict_completion))).cuda()
        ad_frequency = torch.zeros((2, len(self.ad_mmse_frequency_dict_completion))).cuda()

        keys = sorted(self.cn_mmse_frequency_dict_completion.keys())
        for i, item in enumerate(keys):
            cn_frequency[0][i] = item
            cn_frequency[1][i] = self.cn_mmse_frequency_dict_completion[item]
        cn_frequency[1] = cn_frequency[1] / (torch.sum(cn_frequency[1]) * 0.01)

        keys = sorted(self.ad_mmse_frequency_dict_completion.keys())
        for i, item in enumerate(keys):
            ad_frequency[0][i] = item
            ad_frequency[1][i] = self.ad_mmse_frequency_dict_completion[item]
        ad_frequency[1] = ad_frequency[1] / (torch.sum(ad_frequency[1]) * 0.01)

        return cn_frequency, ad_frequency

    @staticmethod
    def completion(mmse_frequency_dict):
        mmse_frequency_dict_completion = dict()
        keys = sorted(mmse_frequency_dict.keys())
        for i, item in enumerate(keys):
            if i == 0:
                continue
            mmse_frequency_dict_completion[float(item)] = mmse_frequency_dict[item]
            for j in range(100):
                new_key = keys[i - 1] + ((keys[i] - keys[i - 1]) / 100) * (j + 1)
                new_value = mmse_frequency_dict[keys[i - 1]] + (
                        (mmse_frequency_dict[keys[i]] - mmse_frequency_dict[keys[i - 1]]) / 100) * (j + 1)
                mmse_frequency_dict_completion[new_key] = new_value
        return mmse_frequency_dict_completion

    @staticmethod
    def reg_loss(reg_score, score, frequency):
        start = score if score <= reg_score else reg_score
        end = reg_score if score <= reg_score else score
        frequency_diff = torch.tensor(0.0).cuda()
        for i in range(frequency.shape[1]):
            if start <= frequency[0][i] <= end:
                frequency_diff += frequency[1][i] * 0.01
            if frequency[0][i] > end:
                break
        return frequency_diff

    @staticmethod
    def forward(ctx, reg_output, demors, frequency, labels):
        cn_frequency, ad_frequency = frequency
        reg_output.squeeze(dim=1)
        con_reg_group_loss = torch.zeros_like(reg_output, requires_grad=True).cuda()

        for i in range(reg_output.shape[0]):
            if labels[i] == 0:
                con_reg_group_loss[i] = ConRegGroupLoss.reg_loss(reg_output[i], demors[i], cn_frequency)
            else:
                con_reg_group_loss[i] = ConRegGroupLoss.reg_loss(reg_output[i], demors[i], ad_frequency)
        con_reg_group_loss = torch.pow(con_reg_group_loss, 2)

        ctx.save_for_backward(reg_output, demors, cn_frequency, ad_frequency, labels, con_reg_group_loss)

        return con_reg_group_loss

    @staticmethod
    def backward(ctx, grad_output):
        reg_output, demors, cn_frequency, ad_frequency, labels, con_reg_group_loss = ctx.saved_tensors
        grad_reg_output = grad_demors = grad_frequency = grad_labels = None

        if ctx.needs_input_grad[0]:
            grad_reg_output = torch.zeros((reg_output.shape[0], 1), requires_grad=True).cuda()
            for i in range(reg_output.shape[0]):
                if labels[i] == 0:
                    frequency_dict = cn_frequency
                else:
                    frequency_dict = ad_frequency

                if frequency_dict[0][0] > reg_output[i]:
                    grad_reg_output[i][0] = grad_reg_output[i][0] - frequency_dict[1][0]
                    continue
                elif frequency_dict[0][-1] < reg_output[i]:
                    grad_reg_output[i][0] = grad_reg_output[i][0] + frequency_dict[1][-1]
                    continue
                last = torch.tensor(0.0).cuda()

                for j in range(frequency_dict.shape[1]):
                    if frequency_dict[0][j] > reg_output[i]:
                        if reg_output[i] < demors[i]:
                            grad_reg_output[i][0] = grad_reg_output[i][0] - \
                                                        (last + frequency_dict[1][j]) * con_reg_group_loss[i]
                        else:
                            grad_reg_output[i][0] = grad_reg_output[i][0] + \
                                                        (last + frequency_dict[1][j]) * con_reg_group_loss[i]
                        break
                    last = frequency_dict[1][j]
        return grad_reg_output, grad_demors, grad_frequency, grad_labels
