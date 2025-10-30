import torch
from torch import nn


class SyMRIT2WParamLayerClamp(nn.Module):
    def __init__(self):
        super().__init__()
        # Unconstrained parameters
        self.TSAT = nn.Parameter(torch.tensor(5000.0))
        self.TE = nn.Parameter(torch.tensor(50.0))
        self.TI = nn.Parameter(torch.tensor(2000.0))

    def forward(self, t1, t2, pd):
        # Clamp parameters to valid ranges (preserves gradients)
        TSAT = self.TSAT.clamp(10, 5000)
        TE = self.TE.clamp(1, 100)
        TI = self.TI.clamp(1000, 3500)

        return (
            torch.abs(pd) *
            torch.exp(-TSAT / (t1 + 1e-5)) *
            torch.exp(-TE / (t2 + 1e-5)) *
            (1 - 2 * torch.exp(-TI / (t1 + 1e-5)))
        )

class SyMRIT2WParamLayerSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        # Raw parameters (unconstrained, will be passed through sigmoid)
        self.raw_TE = nn.Parameter(torch.tensor(0.0))    # maps to [1, 100]
        self.raw_TI = nn.Parameter(torch.tensor(0.0))    # maps to [1000, 3500]
        self.raw_TSAT = nn.Parameter(torch.tensor(0.0))  # maps to [400, 10000]

    def forward(self, t1, t2, pd):
        # Apply sigmoid reparameterization to enforce bounds
        TE = 1 + 99 * torch.sigmoid(self.raw_TE)
        TI = 1000 + 2500 * torch.sigmoid(self.raw_TI)
        TSAT = 400 + 9600 * torch.sigmoid(self.raw_TSAT)

        return (
            torch.abs(pd) *
            torch.exp(-TSAT / (t1*1000 + 1e-5)) *
            torch.exp(-TE / (t2*1000 + 1e-5)) *
            (1 - 2 * torch.exp(-TI / (t1*1000 + 1e-5)))
        )

class SyMRIPSIRParamLayerSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        # Raw parameters (unconstrained, will be passed through sigmoid)
        self.raw_TR = nn.Parameter(torch.tensor(0.0))    # maps to [0, 10000]
        self.raw_TI = nn.Parameter(torch.tensor(0.0))    # maps to [200, 3000]
        self.raw_TE = nn.Parameter(torch.tensor(0.0))  # maps to [0, 100]
        self.eps = 1e-10

    def forward(self, t1, t2):
        # Apply sigmoid reparameterization to enforce bounds
        TE = 0 + 100 * torch.sigmoid(self.raw_TE)
        TI = 200 + 2800 * torch.sigmoid(self.raw_TI)
        TR = 0 + 10000 * torch.sigmoid(self.raw_TR)
        t1_safe = t1.clamp(min=self.eps)
        t2_safe = t2.clamp(min=self.eps)
        psir = (1 - 2 * torch.exp(-TI / (t1_safe*1000)) + torch.exp(-TR / (t1_safe*1000))) * torch.exp(-TE / (t2_safe*1000))

        zero_mask = (t1 == 0) | (t2 == 0)
        psir = psir.masked_fill(zero_mask, 0.0)

        return psir

class SyMRIPSIRParamLayerLog(nn.Module):
    def __init__(self):
        super().__init__()
        # Raw parameters (unconstrained, will be passed through sigmoid)
        self.raw_TR = nn.Parameter(torch.tensor(2.85))    # maps to [1, 10000]
        self.raw_TI = nn.Parameter(torch.tensor(-0.66))    # maps to [200, 3000]
        self.raw_TE = nn.Parameter(torch.tensor(0.))  # maps to [1, 100]
        self.eps = 1e-10

    def forward(self, t1, t2):
        # Apply sigmoid reparameterization to enforce bounds
        TR = torch.exp(torch.log(torch.tensor(1)) +
                  torch.sigmoid(self.raw_TR) * (torch.log(torch.tensor(10000)) - torch.log(torch.tensor(1))))
        TI = torch.exp(torch.log(torch.tensor(200)) +
                  torch.sigmoid(self.raw_TI) * (torch.log(torch.tensor(3000)) - torch.log(torch.tensor(200))))
        TE = torch.exp(torch.log(torch.tensor(1)) +
                  torch.sigmoid(self.raw_TE) * (torch.log(torch.tensor(100)) - torch.log(torch.tensor(1))))
        t1_safe = t1.clamp(min=self.eps)
        t2_safe = t2.clamp(min=self.eps)
        psir = (1 - 2 * torch.exp(-TI / (t1_safe*1000)) + torch.exp(-TR / (t1_safe*1000))) * torch.exp(-TE / (t2_safe*1000))

        zero_mask = (t1 == 0) | (t2 == 0)
        psir = psir.masked_fill(zero_mask, 0.0)

        return psir

class SyMRIGREParamLayerSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        # Raw parameters (unconstrained, will be passed through sigmoid)
        self.raw_TE = nn.Parameter(torch.tensor(0.0))    # maps to [0, 100]
        self.raw_TR = nn.Parameter(torch.tensor(0.0))    # maps to [0, 10000]

    def forward(self, t1, t2, pd):
        # Apply sigmoid reparameterization to enforce bounds
        TE = 0 + 100 * torch.sigmoid(self.raw_TE)
        TR = 0 + 10000 * torch.sigmoid(self.raw_TR)

        return pd * ((1 - torch.exp(-TR/t1/1000))*torch.exp(-TE/t2/1000))

class SyMRIT1WMP2RAGEParamLayerSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        # Raw parameters (unconstrained, will be passed through sigmoid)
        self.raw_TI = nn.Parameter(torch.tensor(0.0))    # maps to [10, 3500]
        self.eps = 1e-10

    def forward(self, t1):
        # Apply sigmoid reparameterization to enforce bounds
        TI = 10 + 3490 * torch.sigmoid(self.raw_TI)
        t1_safe = t1.clamp(min=self.eps)
        t1w = 1-2*torch.exp(-TI/t1_safe/1000)
        zero_mask = (t1 == 0)
        t1w = t1w.masked_fill(zero_mask, 0.0)

        return t1w

class SyMRIT1WMP2RAGEParamLayerLog(nn.Module):
    def __init__(self):
        super().__init__()
        # Raw parameters (unconstrained, will be passed through sigmoid)
        self.raw_TI = nn.Parameter(torch.tensor(1.4))    # maps to [10, 3500]
        self.eps = 1e-10

    def forward(self, t1):
        # Apply sigmoid reparameterization to enforce bounds
        TI = torch.exp(torch.log(torch.tensor(10)) +
                  torch.sigmoid(self.raw_TI) * (torch.log(torch.tensor(3500)) - torch.log(torch.tensor(10))))
        t1_safe = t1.clamp(min=self.eps)
        t1w = 1-2*torch.exp(-TI/t1_safe/1000)
        zero_mask = (t1 == 0)
        t1w = t1w.masked_fill(zero_mask, 0.0)

        return t1w

class SyMRIDIRParamLayerLog(nn.Module):
    def __init__(self):
        super().__init__()
        # Raw parameters (unconstrained, will be passed through sigmoid)
        self.raw_TI1 = nn.Parameter(torch.tensor(1.5))    # maps to [10, 3500]
        self.raw_TI2 = nn.Parameter(torch.tensor(0.1))    # maps to [10, 3500]
        self.raw_TE = nn.Parameter(torch.tensor(1.2))    # maps to [10, 3500]
        self.raw_TR = nn.Parameter(torch.tensor(3.0))    # maps to [10, 3500]
        self.eps = 1e-10

    def forward(self, t1,t2,pd):
        # Apply sigmoid reparameterization to enforce bounds
        TI_1 = torch.exp(torch.log(torch.tensor(500)) +
                  torch.sigmoid(self.raw_TI1) * (torch.log(torch.tensor(6000)) - torch.log(torch.tensor(500))))
        TI_2 = torch.exp(torch.log(torch.tensor(100)) +
                  torch.sigmoid(self.raw_TI2) * (torch.log(torch.tensor(2000)) - torch.log(torch.tensor(100))))
        TE = torch.exp(torch.log(torch.tensor(1)) +
                  torch.sigmoid(self.raw_TE) * (torch.log(torch.tensor(400)) - torch.log(torch.tensor(1))))
        TR = torch.exp(torch.log(torch.tensor(100)) +
                  torch.sigmoid(self.raw_TR) * (torch.log(torch.tensor(20000)) - torch.log(torch.tensor(100))))
        t1_safe = t1.clamp(min=self.eps)
        t2_safe = t2.clamp(min=self.eps)
        pd_safe = pd.clamp(min=self.eps)

        dir = torch.abs((pd_safe) * ( 1 - 2*torch.exp(-TI_2/t1_safe/1000) + 2*torch.exp(-(TI_1+TI_2)/t1_safe/1000) - torch.exp(-TR/t1_safe/1000) )) * (torch.exp(-TE/t2_safe/1000))
        zero_mask = (t1 == 0) | (t2 == 0) | (pd == 0)
        dir = dir.masked_fill(zero_mask, 0.0)

        return dir

class SyMRIT2WFLAREParamLayerLog(nn.Module):
    def __init__(self):
        super().__init__()
        # Raw parameters (unconstrained, will be passed through sigmoid)
        self.raw_TE = nn.Parameter(torch.tensor(1.74))    # maps to [1, 100]
        self.raw_TI = nn.Parameter(torch.tensor(1.6))    # maps to [1000, 3500]
        self.raw_TSAT = nn.Parameter(torch.tensor(0.78))    # maps to [400, 10000]
        self.eps = 1e-10

    def forward(self, t1, t2, pd):
        # Apply sigmoid reparameterization to enforce bounds
        TE = torch.exp(torch.log(torch.tensor(1)) +
                  torch.sigmoid(self.raw_TE) * (torch.log(torch.tensor(200)) - torch.log(torch.tensor(1))))
        TI = torch.exp(torch.log(torch.tensor(100)) +
                  torch.sigmoid(self.raw_TI) * (torch.log(torch.tensor(6000)) - torch.log(torch.tensor(100))))
        TSAT = torch.exp(torch.log(torch.tensor(10)) +
                  torch.sigmoid(self.raw_TSAT) * (torch.log(torch.tensor(1000)) - torch.log(torch.tensor(10))))
        t1_safe = t1.clamp(min=self.eps)
        t2_safe = t2.clamp(min=self.eps)
        pd_safe = pd.clamp(min=self.eps)

        t2w = torch.abs(pd_safe*torch.exp(-TSAT/t1_safe/1000)*torch.exp(-TE/t2_safe/1000)*(1-2*torch.exp(-TI/t1_safe/1000)))
        zero_mask = (t1 == 0) | (t2 == 0) | (pd == 0)
        t2w = t2w.masked_fill(zero_mask, 0.0)

        return t2w

class SyMRIdSIRParamLayerLog(nn.Module):
    def __init__(self):
        super().__init__()
        # Raw parameters (unconstrained, will be passed through sigmoid)
        self.raw_TIi = nn.Parameter(torch.tensor(1.44))    # maps to [1, 100]
        self.raw_TIs = nn.Parameter(torch.tensor(1.1))    # maps to [1000, 3500]
        self.eps = 1e-10

    def forward(self, t1, t2, pd):
        # Apply sigmoid reparameterization to enforce bounds
        TIi = torch.exp(torch.log(torch.tensor(1)) +
                  torch.sigmoid(self.raw_TIi) * (torch.log(torch.tensor(4000)) - torch.log(torch.tensor(1))))
        TIs = torch.exp(torch.log(torch.tensor(1)) +
                  torch.sigmoid(self.raw_TIs) * (torch.log(torch.tensor(4000)) - torch.log(torch.tensor(1))))
        t1_safe = t1.clamp(min=self.eps)

        dsir = (torch.abs(1 - 2 * torch.exp(-TIs / t1_safe/1000)) - torch.abs(1 - 2 * torch.exp(-TIi / t1_safe/1000))) / (torch.abs(1 - 2 * torch.exp(-TIs / t1_safe/1000)) + torch.abs(1 - 2 * torch.exp(-TIi / t1_safe/1000)))
        zero_mask = (t1 == 0)
        dsir = dsir.masked_fill(zero_mask, 0.0)

        return dsir

class SyMRIR2ParamLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-10

    def forward(self, t2):
        # Apply sigmoid reparameterization to enforce bounds
        t2_safe = t2.clamp(min=self.eps)
        r2 = 1 / t2_safe
        zero_mask = (t2 == 0)
        r2 = r2.masked_fill(zero_mask, 0.0)

        return r2