is_amp = True
import warnings

warnings.filterwarnings('ignore')
from torch.nn.functional import normalize
from dataset import *
from augmentation import *
import balanaced_mse
from upernet import *
import matplotlib.pyplot as plt


def balanced_mse_loss(y_pred, y_true, sigma_noise=1.0):
    """
    Balanced MSE loss implementation using Batch-based Monte-Carlo (BMC) approach.
    Parameters:
    - y_pred: Predictions from the model.
    - y_true: Ground truth labels.
    - sigma_noise: Noise scale, sigma_noise^2 is used as the temperature coefficient tau.
    Returns:
    - Loss: Computed Balanced MSE loss.
    """
    tau = 2 * (sigma_noise ** 2)
    mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
    exp_mse_loss = torch.exp(-mse_loss / tau)

    # Sum of exp_mse_loss over the batch for normalization
    sum_exp_mse_loss = torch.sum(exp_mse_loss, dim=0)

    # Balanced MSE loss calculation
    balanced_mse = -torch.log(exp_mse_loss / sum_exp_mse_loss)

    return balanced_mse.mean()

def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    target=target.view(pred.shape[0],1)
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0],device='cuda:0'))     # contrastive-like loss
    loss = loss * (2 * noise_var) # optional: restore the loss scale, 'detach' when noise is learnable
    # loss=loss.detach()
    return loss
class GatedAttention(nn.Module):
    def load_pretrain(self, ):
        a=1
    def __init__(self,arg):
        super(GatedAttention, self).__init__()
        self.arg=arg
        self.ResFeature=arg.Feature_dim
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.ResFeature, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )

    def forward(self, x):
        values=x['values']
        Y=x['label']
        batch_size=len(x['index'])
        # H = values.squeeze(0)
        H = self.feature_extractor_part2(values)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = F.softmax(A, dim=1)  # softmax over K
        A = torch.transpose(A, 2, 1)# ATTENTION_BRANCHESxK

        Z = torch.matmul(A, H).view(batch_size,-1)  # ATTENTION_BRANCHESxM
        Y_prob = self.classifier(Z)
        # Y_prob = self.classifier(Z)[:,1].view(batch_size,1)
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        if self.arg.bmse:
            if self.arg.bmse == 'gai':
                criterion = balanaced_mse.GAILoss(self.arg.init_noise_sigma, self.arg.gmm)
            elif self.arg.bmse == 'bmc':
                criterion = balanaced_mse.BMCLoss(self.arg.init_noise_sigma)
        output = {}
        if 'loss' in self.output_type:
            # Y = Y.float()
            Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
            output['loss'] = criterion(Y_prob, Y)
        if 'inference' in self.output_type:
            output['AttenWeight'] = A.view(batch_size,-1).cpu().detach().numpy()
            output['probability'] = Y_prob
            output['FeatureWeight']=A_U.cpu().detach().numpy()
        return output




