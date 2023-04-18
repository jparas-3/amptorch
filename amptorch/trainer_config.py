from typing import List, Dict, Union, Optional, Sequence
from dataclasses import dataclass
from abc import ABC
from collections.abc import MutableMapping
from datetime import datetime
import os
from ase import Atoms
import torch


class DictEmulator(MutableMapping):
    """
    Abstract base class for emulating a dictionary, inherited by all the config dataclasses
    """
    def __post_init__(self):
        """
        If an optional field that doesn't have a default value is not passed in, delete that field
        :return: None
        """
        for key in list(self.__dict__):
            if self.__getitem__(key) is None:
                self.__delitem__(key)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __iter__(self):
        return iter(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def get(self, key, value=None):
        return self.__dict__.get(key, value)

    def __len__(self):
        return len(self.__dict__)


@dataclass
class FingerprintParams(DictEmulator, ABC):
    """
    Abstract base class for fp_params
    """
    cutoff: int


@dataclass
class GMPsParams(FingerprintParams):
    """
    Dataclass to represent the Gaussian Momenta Parameters (GMPs) dictionary object as input to Config class fp_params.

    Attributes:
    -----------
    orders : List[int]
        List of integer orders of the spherical harmonics functions. For example, [0, 1, 2, 3] would account up to order 3 from order 0.
    sigmas : List[float]
        List of float values of the widths of the Gaussian basis functions.
    atom_gaussians : Dict[str], optional
        Dictionary mapping element symbols to paths of the corresponding pseudodensity files containing Gaussian basis functions for each element.
    square : bool, optional
        Boolean flag indicating whether to square the solid harmonics functions for smoothness in gradient calculation.
        Default is True.
    solid_harmonics : bool, optional
        Boolean flag indicating whether to use solid harmonics functions instead of real spherical harmonics. Default is True.
    """

    orders: List[int]
    sigmas: List[float]
    atom_gaussians: Dict[str, str]
    square: bool = True
    solid_harmonics: bool = True

    # def load_atom_gaussians(self, directory: str) -> None:
    #     for element in self.atom_gaussians:
    #         filename = os.path.join(directory, f"{element}_pseudodensity.g")
    #         self.atom_gaussians[element] = filename


@dataclass
class G2Params(DictEmulator):
    """
    Dataclass representing parameters for the G2 component of the gaussian symmetry function.

    Attributes:
    -----------
    etas: List[float]
        A list of the exponents used in the G2 function.
    rs_s: List[float]
        A list of radial offsets used in the G2 function.
    """

    etas: Sequence[float]
    rs_s: Sequence[float]


@dataclass
class G4Params(DictEmulator):
    """
    Dataclass representing parameters for the G2 component of the gaussian symmetry function.

    Attributes:
    -----------
    etas: List[float]
        A list of exponents used in the G4 function.
    zetas: List[float]
        A list of zetas used in the G4 function.
    gammas: List[float]
        A list of gammas used in the G4 function.
    """

    etas: Sequence[float]
    zetas: Sequence[float]
    gammas: Sequence[float]


@dataclass
class GaussianParams(FingerprintParams):
    """
    Dataclass for Gaussian Symmetry Function with G2 and G4 with a cutoff.

    Attributes:
    -----------
    gaussian : Dict
        A dictionary of Gaussian types (G2 and G4) and their parameters.

    cutoff : float
        The cutoff value.
    """

    G2: G2Params
    G4: G4Params

    # def __init__(self, gaussian, cutoff):
    #     self.gaussian = gaussian
    #     self.cutoff = cutoff
    #
    # @classmethod
    # def from_dict(cls, params_dict):
    #     gaussian_params = params_dict.get("gaussian", {})
    #     g2_params_dict = gaussian_params.get("G2", {})
    #     g4_params_dict = gaussian_params.get("G4", {})
    #     g2_params = G2Params(
    #         etas=g2_params_dict.get("etas", []), rs_s=g2_params_dict.get("rs_s", [])
    #     )
    #     g4_params = G4Params(
    #         etas=g4_params_dict.get("etas", []),
    #         zetas=g4_params_dict.get("zetas", []),
    #         gammas=g4_params_dict.get("gammas", []),
    #     )
    #     return cls(
    #         gaussian={"G2": g2_params, "G4": g4_params},
    #         cutoff=params_dict.get("cutoff", 6),
    #     )


@dataclass
class ConfigModel(DictEmulator):
    """
    Configuration to define the atomistic neural network model configuration.

    Attributes:
    ------------
        name (str): The model to be used for atomistic neural network force field, "SingleNN" or "BPNN". (default:  "SingleNN")

        num_layers (int): No. of hidden layers

        num_nodes (int): No. of nodes per layer

        get_forces (bool): Whether to compute per-atom forces for force training. Set "force_coefficient" in "Optim" accordingly. (default: True)

        batchnorm (bool): Enable batch-normalization (default:False)

        activation (object): Activation function. Any activation supported by torch. (default: nn.Tanh)

        **custom_args: Any additional arguments used to customize existing/new models
    """

    num_layers: int
    num_nodes: int
    name: Optional[str] = "SingleNN"
    get_forces: Optional[bool] = True
    batchnorm: Optional[bool] = False
    activation: Optional[object] = torch.nn.Tanh
    custom_args: Optional[Dict[str, Union[int, bool, object]]] = None


@dataclass
class ConfigOptim(DictEmulator):
    """
    Configuration to define the resources, and setting for neural network optimization.

    Attributes:
    ------------
    gpus (int): No. of gpus to use, 0 for cpu (default: 0)

    force_coefficient (float): If force training, coefficient to weight the force component by (default: 0)

    lr (float): Initial learning rate (default: 1e-1)

    batch_size (int): Batch size (default: 32)

    epochs (int): Max training epochs (default: 100)

    optimizer (object): Training optimizer (default: torch.optim.Adam)

    loss_fn (object): Loss function to optimize (default: CustomLoss)

    loss (str): Control loss function criterion, "mse" or "mae" (default: "mse")

    metric (str): Metrics to be reported by, "mse" or "mae" (default: "mae")

    cp_metric (str): Property based on which the model is saved. "energy" or "forces" (default: "energy")

    scheduler (dict): Learning rate scheduler to use
        {"policy": "StepLR", "params": {"step_size": 10, "gamma": 0.1}}
    """

    gpus: Optional[int] = 0
    force_coefficient: Optional[float] = 0
    lr: Optional[float] = 1e-1
    batch_size: Optional[int] = 32
    epochs: Optional[int] = 100
    optimizer: Optional[object] = torch.optim.Adam
    loss: Optional[str] = "mse"
    metric: Optional[str] = "mae"
    cp_metric: Optional[str] = "energy"
    scheduler: Optional[object] = None


@dataclass
class ConfigDataset(DictEmulator):
    """
    Configuration to define the dataset used for training, and featurization scheme to be used. Featurization default to Guassian Multipole Fingerprinting.

    Attributes:
    ------------
    raw_data (str or List[ase.Atoms]]): Path to ASE trajectory or database, or list of Atoms objects.

    lmdb_path (Optional: str): Path to LMDB database file for dataset too large to fit in memory, if raw_data is not provided.

    val_split (float): Proportion of training set to use for validation.

    elements (Optional: List[str]): List of unique elements in dataset, optional. Example: ["H", "O"] for water dataset.

    fp_scheme (str): Fingerprinting scheme to feature dataset, "gmpordernorm" or "gaussian" (default: "gaussian").

    fp_params (dict): Fingerprint parameters, dataclass "GMP_params" or "SF_params".

    cutoff_params (Optional, dict): Cutoff function for Gaussian fingerprinting scheme only - polynomial or cosine,
        polynomial - {"cutoff_func": "Polynomial", "gamma": 2.0},
        cosine - {"cutoff_func": "Cosine"}.

    save_fps (bool): Write calculated fingerprints to disk (default: True).

    scaling (Optional[Dict[str, Any]]): Feature scaling scheme, normalization or standardization,
        normalization (scales features between "range") - {"type": "normalize", "range": (0, 1)},
        standardization (scales data to mean=0, stdev=1) - {"type": "standardize"}.
    """

    raw_data: Union[str, List[Atoms]] = None
    lmdb_path: str = None
    val_split: float = None
    elements: Optional[List[str]] = None
    fp_scheme: Optional[str] = "gaussian"
    fp_params: Dict[str, GaussianParams] = None
    cutoff_params: Optional[object] = None
    save_fps: Optional[bool] = True
    scaling: Optional[dict] = None


@dataclass
class ConfigCmd(DictEmulator):
    """
    Configuration for the extra commands and idenfiers.

    Attributes:
    -----------
        debug : Optional, bool
            If True, enables debug mode, which does not write/save checkpoints/results. Defaults to False.

        dtype : Optional, type
            PyTorch level of precision. Defaults to torch.DoubleTensor.

        run_dir : Optional, str
            Path to the directory where logs are to be saved. Defaults to "./".

        seed : Optional, int
            Random seed to use. If not specified, a random seed of 0 will be used to ensure consistency.

        identifier : Optional, str
            Unique identifier for the experiment. If not specified, the current time will be used.

        verbose : Optional, bool
            If True, print training scores. Defaults to True.

        logger : Optional, bool
            If True, log results to Weights and Biases (https://www.wandb.com/). A free account is necessary to view and log results.
    """

    debug: Optional[bool] = False
    dtype: Optional[object] = torch.DoubleTensor
    run_dir: Optional[str] = "./"
    seed: Optional[int] = 0
    identifier: Optional[str] = str(datetime.now())
    verbose: Optional[bool] = True
    logger: Optional[bool] = False


@dataclass
class Config(DictEmulator):
    """
    Input configuration for trainer class to start the training of NNFF.

    Attributes:
    -----------
    model: Model dataclass object or dictionary. Specifying the configuration of neural network force field model.

    optim: Optim dataclass object or dictionary. Specifying the optimizer and resources for optimization.

    dataset: Dataset dataclass object or dictionary. Specifying the dataset used for training and validation, and the fingerprinting scheme.

    cmd: Extra commands for debugging, running directory, identifier.
    """

    model: ConfigModel
    optim: ConfigOptim
    dataset: ConfigDataset
    cmd: ConfigCmd
