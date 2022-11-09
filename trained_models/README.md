* Basic UNETR + Deep Supervision + Self-Distillation with Shape Priors:Save the last model here while training with `train_SelfDistil_DistMaps.py` as `last_CT_challenge_data_SelfDistil_DistMaps.pth`.

* Basic UNETR + Deep Supervision + Self-Distillation:Save the last model here while training with `train_SelfDistil_Original.py` as `last_CT_challenge_data_SelfDistil_Original.pth`.

* Basic UNETR + Deep Supervision: Save the last model here while training with `train_DeepSuperOnly.py` as `last_CT_challenge_data_DeepSuperOnly.pth`.

* Basic UNETR: Save the last model here while training with `train_basicUNETR.py` as `last_CT_challenge_data_BasicUnetr.pth`.

* Load these last models as checkpoint in `test.py`, in case you need to evaluate on the validation set with the trained model at the last epoch.

* NOTE: The last models could not be uploaded due to size limits.

