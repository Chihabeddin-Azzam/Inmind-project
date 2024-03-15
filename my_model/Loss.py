import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import intersection_over_union

class Loss(nn.Module):
    def __init__(self, S=7, B=2, C=3):
        super(Loss, self).__init__()
        self.S = S  # Split size of the image
        self.B = B  # Number of bounding boxes per grid cell
        self.C = C  # Number of classes

        # Weights for different components of the loss function
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

        # Define the loss functions
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, predictions, targets):
        """
        Calculate YOLO loss
        Args:
            predictions: tensor (batch_size, S*S*(B*5+C)) - predicted output from the model
            targets: tensor (batch_size, S*S*(B*5+C)) - ground truth annotations

        Returns:
            loss: tensor - total YOLO loss
        """

        # Reshape predictions and targets to match the shape of the output
        predictions = predictions.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        targets = targets.reshape(-1, self.S, self.S, self.B * 5 + self.C)

        # Split predictions into bounding box coordinates, objectness scores, and class predictions
        pred_boxes = predictions[..., :self.B * 5].reshape(-1, self.S, self.S, self.B, 5)
        pred_obj = predictions[..., self.B * 5:self.B * 5 + 1]
        pred_cls = predictions[..., self.B * 5 + 1:]

        # Split targets into bounding box coordinates, objectness scores, and class annotations
        true_boxes = targets[..., :self.B * 5].reshape(-1, self.S, self.S, self.B, 5)
        true_obj = targets[..., self.B * 5:self.B * 5 + 1]
        true_cls = targets[..., self.B * 5 + 1:]

        # Calculate the objectness loss (binary cross-entropy loss)
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, true_obj, reduction='sum')

        # Calculate the coordinates loss (MSE loss)
        coord_loss = self.mse_loss(torch.flatten(pred_boxes[..., :4], end_dim=-2),
                                    torch.flatten(true_boxes[..., :4], end_dim=-2))

        # Calculate the confidence loss (MSE loss)
        noobj_loss = self.mse_loss(torch.flatten(pred_obj, end_dim=-2),
                                    torch.flatten(true_obj, end_dim=-2))

        # Calculate the class loss (MSE loss)
        class_loss = self.mse_loss(torch.flatten(pred_cls, end_dim=-2),
                                    torch.flatten(true_cls, end_dim=-2))

        # Total YOLO loss
        loss = (self.lambda_coord * coord_loss +
                obj_loss +
                self.lambda_noobj * noobj_loss +
                class_loss)

        return loss
