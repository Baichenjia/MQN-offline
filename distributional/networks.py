import numpy as np
import torch
import torch.nn as nn


class QuantileMlp(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            embedding_size=64,
            num_quantiles=32,
            layer_norm=True,
            **kwargs,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        # hidden_sizes[:-2] MLP base
        # hidden_sizes[-2] before merge
        # hidden_sizes[-1] before output

        self.base_fc = []
        last_size = input_size
        for next_size in hidden_sizes[:-1]:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)
        self.num_quantiles = num_quantiles
        self.embedding_size = embedding_size
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, last_size),
            nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )
        self.merge_fc = nn.Sequential(
            nn.Linear(last_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)
        self.const_vec = torch.from_numpy(np.arange(1, 1 + self.embedding_size)).float().to('cuda')

    def forward(self, state, action, tau):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """
        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        h = self.merge_fc(h)  # (N, T, C)
        output = self.last_fc(h).squeeze(-1)  # (N, T)
        return output

    def penultimate_layer(self, state, action):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """
        presum_tau = torch.zeros(len(action), self.num_quantiles) + 1. / self.num_quantiles
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = torch.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
            tau_hat = tau_hat.to(state.device)

        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)  # (N, C)

        x = torch.cos(tau_hat.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        h = self.merge_fc(h)  # (N, T, C)
        return h


class QuantileDueling(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            embedding_size=64,
            num_quantiles=32,
            layer_norm=True,
            dim_state=None,
            **kwargs,
    ):
        super().__init__()

        self.base_fc = []
        last_size = input_size
        for next_size in hidden_sizes[:-1]:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)
        self.num_quantiles = num_quantiles
        self.embedding_size = embedding_size
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, last_size),
            nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )
        self.merge_fc = nn.Sequential(
            nn.Linear(last_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.last_fc = nn.Sequential(nn.Linear(hidden_sizes[-1], 1), nn.ReLU())
        self.const_vec = torch.from_numpy(np.arange(1, 1 + self.embedding_size)).float().to('cuda')
        self.leftmultiply = torch.ones(self.num_quantiles, self.num_quantiles).triu(diagonal=0).T.cuda()

        # baseline net
        self.baseline_net = nn.Sequential(
            nn.Linear(input_size, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, state, action, tau, K=None):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)  (256, 32)
        """
        if K is not None:
            tau = tau[:, :K+1]
        else:
            K = self.num_quantiles-1

        hs = torch.cat([state, action], dim=1)
        hs = self.base_fc(hs)                    # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)
        x = self.tau_fc(x)                       # (N, T, C)

        h = torch.mul(x, hs.unsqueeze(-2))       # (N, T, C)
        h = self.merge_fc(h)                     # (N, T, C)
        output = self.last_fc(h).squeeze(-1)     # (N, T)       （256, 32）

        #
        batch_size = state.size()[0]
        non_crossing_matrix = self.leftmultiply[:K+1, :K+1].repeat(batch_size, 1, 1) / self.num_quantiles / 10.   # (b, 32, 32)
        output_non_crossing = torch.bmm(non_crossing_matrix, output.view(batch_size, K+1, 1)).squeeze(2)
        assert output_non_crossing.size() == (batch_size, K+1)

        # baseline net
        value_baseline = self.baseline_net(torch.cat([state, action], dim=1))
        output_non_crossing_dueling = output_non_crossing + value_baseline  # (batch_size, 32)

        # if batch_size == 256:
        #     print("tau:", tau.size(), ", K:", K)
        #     print("output:", output.size())
        #     print("non crossing matrix:", non_crossing_matrix.size())
        #     print("non crossing output:", output_non_crossing[5])
        #     print("")

        return output_non_crossing_dueling



# class NDQFN(nn.Module):
#     def __init__(
#             self,
#             hidden_sizes,       # (256, 256)
#             dim_state,
#             dim_action,
#             embedding_size=64,  # (64)
#             num_quantiles=32,
#             layer_norm=True,    # true
#             device='cuda',
#             **kwargs,
#     ):
#         super().__init__()
#         self.layer_norm = layer_norm
#         self.device = device
#         # hidden_sizes[:-2] MLP base
#         # hidden_sizes[-2] before merge
#         # hidden_sizes[-1] before output
#         input_size = dim_state + dim_action
#         self.base_fc = []
#         last_size = input_size
#         for next_size in hidden_sizes[:-1]:
#             self.base_fc += [
#                 nn.Linear(last_size, next_size),
#                 nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
#                 nn.ReLU(inplace=True),
#             ]
#             last_size = next_size
#         self.base_fc = nn.Sequential(*self.base_fc)
#         self.num_quantiles = num_quantiles
#         self.K = num_quantiles
#         self.embedding_size = embedding_size
#         self.tau_fc = nn.Sequential(
#             nn.Linear(embedding_size, last_size),
#             nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
#             nn.Sigmoid(),
#         )
#         self.merge_fc = nn.Sequential(
#             nn.Linear(last_size, hidden_sizes[-1]),
#             nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
#             nn.ReLU(inplace=True),
#         )
#         self.last_fc = nn.Sequential(nn.Linear(hidden_sizes[-1], 1), nn.ReLU())
#         self.const_vec = torch.from_numpy(np.arange(1, 1 + self.embedding_size)).float().to(device)
#
#         # baseline net
#         self.baseline_net = nn.Sequential(
#             nn.Linear(dim_state, 256), nn.LayerNorm(256), nn.ReLU(),
#             nn.Linear(256, 16), nn.ReLU(),
#             nn.Linear(16, 1))
#
#         # leftmultiply
#         self.leftmultiply = torch.ones(self.K+1, self.K+1).triu(diagonal=0).T.to(device)
#         self.tau_supp, self.tau_hat_supp, self.presum_tau_supp = self.get_tau_support(batch_size=1)
#
#     def get_tau_support(self, batch_size):
#         presum_tau = torch.zeros(batch_size, self.K + 2) + 1.0 / self.K  # 1/32,1/32,...   (b, K+2)
#         tau = torch.cumsum(presum_tau, dim=1) - 1.5 / self.K             # -0.5/K, 0.5/K, 1.5/K, ..., (K+0.5)/K   (b, K+2)
#         with torch.no_grad():
#             tau_hat = (tau[:, 1:] + tau[:, :-1]) / 2.                    # (b, K+1)
#         return tau.to(self.device), tau_hat.to(self.device), presum_tau.to(self.device)
#
#     def get_support_quantile(self, state, action):
#         batch_size = state.size()[0]
#         tau_hat = self.tau_hat_supp.repeat(batch_size, 1)
#
#         h = torch.cat([state, action], dim=1)         # (batch_size, 23)
#         h = self.base_fc(h)                           # (batch_size, 256)
#
#         x = torch.cos(tau_hat.unsqueeze(-1) * self.const_vec * np.pi)  # (batch_size, K+1, embedding_size)
#         x = self.tau_fc(x)                            # (batch_size, K+1, 256)
#
#         h = torch.mul(x, h.unsqueeze(-2))             # (batch_size, K+1, 256)
#         h = self.merge_fc(h)                          # (batch_size, K+1, 256)
#
#         output = self.last_fc(h).squeeze(-1)          # (batch_size, K+1)
#
#         # non-crossing constrain
#         non_crossing_matrix = self.leftmultiply.repeat(batch_size, 1, 1) / self.K / 10.  # [/k/10.]
#         output_non_crossing = torch.bmm(non_crossing_matrix, output.view(batch_size, self.K+1, 1)).squeeze()
#         assert output_non_crossing.size() == (batch_size, self.K+1)
#         # print("non-crossing:", output_non_crossing[5])
#
#         # state baseline
#         value_baseline = self.baseline_net(state)
#         output_non_crossing_dueling = output_non_crossing + value_baseline   # (batch_size, 32)
#         # print("output_non_crossing_dueling:", output_non_crossing_dueling[5], output_non_crossing_dueling.size())
#
#         # return output_non_crossing
#         return output_non_crossing_dueling
#
#     def forward(self, state, action, tauk):
#         bsize, dim = tauk.size()[0], tauk.size()[1]  # tauk.size() = (256, 32), or (2560, 1)
#
#         quantiles_fixed = self.get_support_quantile(state, action)  # (batch_size, K+1)
#         assert quantiles_fixed.size() == (bsize, self.K+1)
#
#         with torch.no_grad():
#             b = tauk / (1. / self.K)
#             l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)  # (256, K) or (2560, 1)
#
#             offset = torch.linspace(0, (bsize - 1) * (self.K+1), bsize).view(-1, 1).cuda()  # (256, 1) or (2560, 1)
#             left_index = (offset + l).to(torch.int64)                   # (256, 32) or (2560, 1)
#             right_index = (offset + u).to(torch.int64)                  # (256, 32) or (2560, 1)
#
#         left = torch.index_select(quantiles_fixed.view(-1), dim=0, index=left_index.view(-1))    # (256*32,) or (2560,)
#         right = torch.index_select(quantiles_fixed.view(-1), dim=0, index=right_index.view(-1))  # (256*32,) or (2560,)
#         output = left.view(bsize, dim) * (u.float() - b) + right.view(bsize, dim) * (b - l.float())
#
#         assert output.size() == (bsize, dim)       # (256, 32) or (2560, 1)
#         return output
#
#     def forward_quantile(self, state, action, tauk):
#         # forward a single quantile
#         bsize = tauk.size()[0]
#         assert tauk.size() == (bsize, 1)
#
#         umax = (tauk.max() / (1. / self.K)).ceil().to(torch.int64)
#         tau_hat = self.tau_hat_supp.repeat(bsize, 1)[:, :umax+1]    # (batch_size, u+1)
#         assert tau_hat.size() == (bsize, umax+1)
#
#         h = torch.cat([state, action], dim=1)  # (batch_size, 23)
#         h = self.base_fc(h)  # (batch_size, 256)
#
#         x = torch.cos(tau_hat.unsqueeze(-1) * self.const_vec * np.pi)  # (batch_size, u+1, embedding_size)
#         x = self.tau_fc(x)  # (batch_size, u+1, 256)
#
#         h = torch.mul(x, h.unsqueeze(-2))  # (batch_size, u+1, 256)
#         h = self.merge_fc(h)  # (batch_size, u+1, 256)
#
#         output = self.last_fc(h).squeeze(-1)  # (batch_size, u+1)
#
#         # non-crossing constrain
#         non_crossing_matrix = self.leftmultiply[:umax+1, :umax+1].repeat(bsize, 1, 1) / self.K / 10.  # [/k/10.]
#         output_non_crossing = torch.bmm(non_crossing_matrix, output.view(bsize, umax+1, 1)).squeeze()
#         assert output_non_crossing.size() == (bsize, umax+1)
#
#         # state baseline
#         value_baseline = self.baseline_net(state)
#         quantile_fixed = output_non_crossing + value_baseline  # (batch_size, u+1)
#         assert output_non_crossing.size() == quantile_fixed.size() == (bsize, umax+1)
#         # print("output_non_crossing_dueling:", output_non_crossing_dueling[5], output_non_crossing_dueling.size())
#
#         with torch.no_grad():
#             b = tauk / (1. / self.K)
#             l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)  # (2560, 1)
#
#             offset = torch.linspace(0, (bsize - 1) * (umax+1), bsize).view(-1, 1).cuda()  # (2560, 1)
#             left_index = (offset + l).to(torch.int64)                   # (2560, 1)
#             right_index = (offset + u).to(torch.int64)                  # (2560, 1)
#
#         left = torch.index_select(quantile_fixed.view(-1), dim=0, index=left_index.view(-1))         # (2560,)
#         right = torch.index_select(quantile_fixed.view(-1), dim=0, index=right_index.view(-1))       # (2560,)
#         output = left.view(bsize, 1) * (u.float() - b) + right.view(bsize, 1) * (b - l.float())
#         return output



# class QuantileMlpNonCrossing(nn.Module):
#     def __init__(
#             self,
#             hidden_sizes,
#             output_size,
#             input_size,
#             embedding_size=64,
#             num_quantiles=32,
#             layer_norm=True,
#             **kwargs,
#     ):
#         super().__init__()
#         self.layer_norm = layer_norm
#
#         self.base_fc = []
#         last_size = input_size
#         for next_size in hidden_sizes[:-1]:
#             self.base_fc += [
#                 nn.Linear(last_size, next_size),
#                 nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
#                 nn.ReLU(inplace=True),
#             ]
#             last_size = next_size
#         self.base_fc = nn.Sequential(*self.base_fc)
#         self.num_quantiles = num_quantiles
#         self.embedding_size = embedding_size
#         self.tau_fc = nn.Sequential(
#             nn.Linear(embedding_size, last_size),
#             nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
#             nn.Sigmoid(),
#         )
#         self.merge_fc = nn.Sequential(
#             nn.Linear(last_size, hidden_sizes[-1]),
#             nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
#             nn.ReLU(inplace=True),
#         )
#         # self.last_fc = nn.Linear(hidden_sizes[-1], 1)
#
#         # self.last_fc_weight = nn.Sequential(nn.Linear(hidden_sizes[-1], 1), nn.ReLU(inplace=True))  # 这个系数需要为正
#         self.last_fc_weight = nn.Sequential(nn.Linear(hidden_sizes[-1], 1), nn.Tanh())  # 这个系数需要为正
#         self.last_fc_bias = nn.Linear(hidden_sizes[-1], 1)
#
#         self.const_vec = torch.from_numpy(np.arange(1, 1 + self.embedding_size)).float().to('cuda')
#
#     def forward(self, state, action, tau):
#         """
#         Calculate Quantile Value in Batch
#         tau: quantile fractions, (N, T)  (256, 32)
#         """
#         h = torch.cat([state, action], dim=1)
#         h = self.base_fc(h)                      # (N, C)
#
#         x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)
#         x = self.tau_fc(x)                       # (N, T, C)
#
#         h = torch.mul(x, h.unsqueeze(-2))        # (N, T, C)
#         h = self.merge_fc(h)                     # (N, T, C)
#         output_weight = self.last_fc_weight(h).squeeze(-1) + 1.0    # (N, T)      (256, 32) or (2560, 1)
#
#         # added
#         output_bias = self.last_fc_bias(h).squeeze(-1)  # (N, T)      (256, 32) or (2560, 1)
#         # (256, 32, 32) + (256, 1, 32) = (256, 32, 32)       or      (2560, 1, 1) + (2560, 1, 1) = (2560, 1, 1)
#         non_crossing_out = torch.matmul(tau.unsqueeze(2), output_weight.unsqueeze(1)) + output_bias.unsqueeze(1)
#         output = non_crossing_out.mean(-1)      # (256, 32) or (2560, 1)
#
#         # if state.size()[0] == 2560:
#         #     print("\n\n***", torch.matmul(tau.unsqueeze(2), output_weight.unsqueeze(1)).size(), output_bias.unsqueeze(1).size())
#         #     print("**", output_weight.size(), output_bias.size(), tau.size(), non_crossing_out.size(), output.size())
#         #     print("output:", output[0])
#
#         return output
#
