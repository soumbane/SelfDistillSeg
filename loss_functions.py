from turtle import forward
from typing import Optional, Iterable, Callable, List, Sequence, cast, Set, Any, Union
import warnings
import torch
import torch.nn.functional as F
import numpy as np

from monai.losses import TverskyLoss
from monai.utils import LossReduction

from torchmanager import losses
from torchmanager_core import _raise


class FocalTverskyLoss(TverskyLoss):
    def __init__(
        self, 
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: int = 1,
    ) -> None:

        super().__init__(include_background, to_onehot_y, sigmoid, softmax, other_act, alpha, beta, reduction=LossReduction.NONE)

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        Returns:
            ft_loss: the focal Tversky loss.

        """
        
        tversky_loss = super().forward(input=input, target=target) # Tversky Loss
        ft_loss = tversky_loss ** (1/self.gamma)

        return torch.mean(ft_loss)


class BoundaryLoss(losses.Loss):

    def __init__(self, *args, include_background: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)  
        self.include_background = include_background # whether to include bkg class or not

    def forward(self, input: torch.Tensor, target: Any) -> torch.Tensor:
        """
        Args:
            input: predicted logits (batch_size, num_class, x,y,z)
            target (GT Dist Map Labels): ground-truth dist map labels, shape (batch_size, num_class, x,y,z)
        Returns:
            boundary_loss: the boundary loss (scalar tensor)
        """
                
        pred_logits_out: torch.Tensor = input # (B, num_cls, x,y,z)
        #print("Preds Tensor shape: ", pred_logits_out.shape)

        pred_probs: torch.Tensor = F.softmax(pred_logits_out, dim=1)
        # print("Softmax Probs shape: ", pred_probs.shape)
        
        # predicted softmax probs for foreground classes ONLY
        if not self.include_background:
            pc: torch.Tensor = pred_probs[:, 1:, ...].type(torch.float32) # considering only foreground classes         
            # print("Softmax Probs Foreground ONLY shape: ", pc.shape)

               
        dist: torch.Tensor = target # (B, num_cls, x,y,z)
        # print("Distance Map Tensor shape: ", dist.shape)
        
        # Ground-Truth Distance Maps for foreground classes ONLY
        if not self.include_background:
            dc: torch.Tensor = dist[:, 1:, ...].type(torch.float32)
            # print("Distance Map Foreground ONLY Tensor shape: ", dc.shape)
        
        multipled: torch.Tensor = torch.einsum("bkxyz,bkxyz->bkxyz", pc, dc)
        boundary_loss: torch.Tensor = multipled.mean()
        #print("Boundary Loss shape: ", boundary_loss.shape)
        
        return boundary_loss


# For DiceCE Loss between GT Labels(target="out") [target] and softmax(out_main,out_dec2,out_dec3,out_dec4) [input]
class Self_Distillation_Loss_Dice(losses.MultiLosses):
    def forward(self, input: Union[Sequence[torch.Tensor], torch.Tensor], target: Any) -> torch.Tensor:
        # initilaize
        loss = 0
        l = 0
        
        # Validation Mode
        if isinstance(input, torch.Tensor): return self.losses[0](input, target)
        
        # Training Mode
        # get all losses
        for i, fn in enumerate(self.losses):
            assert isinstance(fn, losses.Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            l = fn(input[i], target)
            loss += l

        del l
        # return loss
        assert isinstance(loss, torch.Tensor), _raise(TypeError("The total loss is not a valid `torch.Tensor`."))
        return loss


Self_Distillation_Loss_Boundary = Self_Distillation_Loss_Dice


# For KL Div Loss
class Self_Distillation_Loss_KL(losses.MultiLosses):

    def __init__(self, *args, include_background: bool = True, T: int = 1, **kwargs) -> None:

        super().__init__(*args, **kwargs)  
        self.include_background = include_background # whether to include bkg class or not
        self.T = T  # divided by temperature (T) to smooth logits before softmax  

    def forward(self, input: Union[Sequence[torch.Tensor], torch.Tensor], target: Any) -> torch.Tensor:
        
        # Validation Mode - Just use the main output (out_main)
        if isinstance(input["out"], torch.Tensor): 
            assert (self.training == False), _raise(TypeError("Should be in validation mode.")) 
            loss = torch.tensor(0, dtype=input["out"].dtype, device=input["out"].device)
            return loss

        ############################################################################################################
        ## Decoder Teacher-Student
        ## For KL Div between softmax(out_dec2) [target/teacher] and log_softmax((out_dec3,out_dec4)) [input/students]
        
        # Training Mode
        target_logits: torch.Tensor = input["out"][3] # out_dec2: Teacher model output (deepest decoder)
                        
        target_logits = target_logits/self.T  # divided by temperature (T) to smooth logits before softmax

        if not self.include_background:
            target_logits = target_logits[:, 1:, ...] # considering only foreground classes
        
        # target softmax probs
        target: torch.Tensor = F.softmax(target_logits, dim=1)  # (B, num_cls, x,y,z): num_cls=7 (since excluding bkg class)
                
        assert isinstance(target, torch.Tensor), _raise(TypeError("Target should be a tensor.")) 

        # initilaize
        loss = 0
        l = 0
        
        # get all KL Div losses between softmax(out_dec2)[target] and log_softmax((out_dec3,out_dec4))[input]
        for i, fn in enumerate(self.losses):
            assert isinstance(fn, losses.Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            
            input_logits_before_log = (input["out"][i+1])/self.T  # (B, num_cls, x,y,z): num_cls=7 (since excluding bkg class): for Decoders - out_dec2 as Teacher and out_dec3, out_dec4 as students

            if not self.include_background:
                input_logits_before_log = input_logits_before_log[:, 1:, ...]  # considering only foreground classes

            log_input = F.log_softmax(input_logits_before_log, dim=1)

            l = fn(log_input, target)

            loss += l * (self.T ** 2)

        ############################################################################################################
        ## Encoder Teacher-Student
        ## For KL Div between softmax(out_enc4) [target/teacher] and log_softmax((out_enc2,out_enc3)) [input/students]
        
        # Training Mode
        target_logits: torch.Tensor = input["out"][6] # out_enc4: Teacher model output (deepest encoder)
                        
        target_logits = target_logits/self.T  # divided by temperature (T) to smooth logits before softmax

        if not self.include_background:
            target_logits = target_logits[:, 1:, ...] # considering only foreground classes
        
        # target softmax probs
        target: torch.Tensor = F.softmax(target_logits, dim=1)  # (B, num_cls, x,y,z): num_cls=7 (since excluding bkg class)
                
        assert isinstance(target, torch.Tensor), _raise(TypeError("Target should be a tensor.")) 

        # initilaize        
        l = 0
        
        # get all KL Div losses between softmax(out_enc4)[target] and log_softmax((out_enc2,out_enc3))[input]
        for i, fn in enumerate(self.losses):
            assert isinstance(fn, losses.Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            
            input_logits_before_log = (input["out"][i+4])/self.T  # (B, num_cls, x,y,z): num_cls=7 (since excluding bkg class): for Encoders - out_enc4 as Teacher and out_enc2, out_enc3 as students

            if not self.include_background:
                input_logits_before_log = input_logits_before_log[:, 1:, ...]  # considering only foreground classes

            log_input = F.log_softmax(input_logits_before_log, dim=1)

            l = fn(log_input, target)

            loss += l * (self.T ** 2)

        ############################################################################################################
        
        del l
        # return loss
        assert isinstance(loss, torch.Tensor), _raise(TypeError("The total loss is not a valid `torch.Tensor`."))
        return loss


# For L2 loss between feature maps/hints dec3_f [target] and feature maps (dec1_f,dec2_f) [input]
class Self_Distillation_Loss_L2(losses.MultiLosses):
    
    def forward(self, input: Union[Sequence[torch.Tensor], torch.Tensor], target: Any) -> torch.Tensor:

        if isinstance(input["out"], torch.Tensor): 
            assert (self.training == False), _raise(TypeError("Should be in validation mode.")) 
            loss = torch.tensor(0, dtype=input["out"].dtype, device=input["out"].device)
            return loss

        target: torch.Tensor = input["out"][6] # dec3_f: Teacher model output (deepest decoder)
        
        assert isinstance(target, torch.Tensor), _raise(TypeError("Target should be a tensor.")) 

        # initilaize
        loss = 0
        l = 0

        # get all L2 losses between feature maps/hints dec3_f[target] and feature maps (dec1_f,dec2_f)[input]
        for i, fn in enumerate(self.losses):
            assert isinstance(fn, losses.Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            l = fn(input["out"][i+4], target)
            loss += l

        del l
        # return loss
        assert isinstance(loss, torch.Tensor), _raise(TypeError("The total loss is not a valid `torch.Tensor`."))
        return loss

        