from pydantic import BaseModel, Field
from typing import List, Union


class InferenceRequest(BaseModel):
    """
    Schema for incoming trajectory prediction requests.
    Expects nested Python lists representing the batches.
    """

    # 3D list: [Batch, Time_past, Features]
    agent: List[List[List[float]]] = Field(
        ..., description="Agent history. Expected shape: [Batch, T_past, 4]"
    )

    # Typed Union for backwards compatibility and clarity
    # [Batch, N, 4] OR [Batch, N, T_past, 4]
    neighbors: Union[List[List[List[float]]], List[List[List[List[float]]]]] = Field(
        ..., description="Surrounding neighboring agents."
    )

    # 4D list: [Batch, Channel, Height, Width]
    map_img: List[List[List[List[float]]]] = Field(
        ..., description="Scene Map crop. Expected shape: [Batch, 3, Height, Width]"
    )


class InferenceResponse(BaseModel):
    """
    Schema for outgoing trajectory predictions.
    """

    # 4D list: [Batch, K, Time_Future, Params]
    trajectories: List[List[List[List[float]]]] = Field(
        ..., description="Predicted modal trajectories. Shape: [Batch, K, T_future, 2]"
    )

    # 2D list: [Batch, K]
    mode_probabilities: List[List[float]] = Field(
        ...,
        description="Softmax probability scores for each predicted mode. Shape: [Batch, K]",
    )
