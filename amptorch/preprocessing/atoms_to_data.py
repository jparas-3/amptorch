import ase.db.sqlite
import ase.io.trajectory
import numpy as np
import torch
from torch_geometric.data import Data

from amptorch.descriptor.descriptor_calculator import DescriptorCalculator

try:
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


class AtomsToData:
    def __init__(
        self,
        descriptor,
        r_energy=False,
        r_forces=False,
        save_fps=True,
        fprimes=True,
        cores=1,
    ):
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.descriptor = descriptor
        self.save_fps = save_fps
        self.fprimes = fprimes
        self.cores = cores

    def convert(
        self, atoms, idx, ref_positions=None,
    ):
        if ref_positions is None:
            ref_positions = atoms.get_positions(wrap=True)

        if weights is None:
            weights = [1.0] * len(ref_positions)

        assert isinstance(ref_positions, np.ndarray)
        assert ref_positions.dtype == "float64"
        assert len(weights) == len(ref_positions)

        descriptor_calculator = DescriptorCalculator(
            images=[atoms],
            ref_positions=[ref_positions],
            descriptor=self.descriptor,
            calc_derivatives=self.fprimes,
            save_fps=self.save_fps,
            cores=self.cores,
            verbose=False,
        )
        self.descriptor_data = descriptor_calculator.prepare_descriptors()

        natoms = len(atoms)
        nrefs = len(ref_positions)
        image_data = self.descriptor_data[0]
        atomic_numbers = torch.LongTensor(atoms.get_atomic_numbers())
        image_fingerprint = torch.tensor(
            image_data["descriptors"], dtype=torch.get_default_dtype()
        )
        weights = torch.tensor(weights, dtype=torch.get_default_dtype())

        # put the minimum data in torch geometric data object
        data = Data(
            fingerprint=image_fingerprint,
            atomic_numbers=atomic_numbers,
            num_nodes=nrefs,
            weights=weights,
        )

        # optionally include other properties
        if self.r_energy:
            energy = atoms.get_potential_energy(apply_constraint=False)
            data.energy = energy
        if self.r_forces:
            forces = torch.tensor(
                atoms.get_forces(apply_constraint=False),
                dtype=torch.get_default_dtype(),
            )
            data.forces = forces
        if self.fprimes:
            fp_prime_val = image_data["descriptor_primes"]["val"]
            fp_prime_row = image_data["descriptor_primes"]["row"]
            fp_prime_col = image_data["descriptor_primes"]["col"]
            fp_prime_size = image_data["descriptor_primes"]["size"]

            indices = np.vstack((fp_prime_row, fp_prime_col))
            torch_indices = torch.LongTensor(indices)
            torch_values = torch.tensor(fp_prime_val, dtype=torch.get_default_dtype())
            # TODO: check how to do double tensor here
            fp_primes = torch.sparse.FloatTensor(
                torch_indices, torch_values, torch.Size(fp_prime_size)
            )

            data.fprimes = fp_primes

        return data

    def convert_all(
        self,
        atoms_collection,
        ref_positions_list=None,
        weights_list=None,
        disable_tqdm=False,
    ):
        """Convert all atoms objects in a list or in an ase.db to graphs.

        Args:
            atoms_collection (list of ase.atoms.Atoms or ase.db.sqlite.SQLite3Database):
            Either a list of ASE atoms objects or an ASE database.

        Returns:
            data_list (list of torch_geometric.data.Data):
            A list of torch geometric data objects containing molecular graph info and properties.
        """

        if isinstance(atoms_collection, list):
            atoms_iter = atoms_collection
        # elif isinstance(atoms_collection, ase.db.sqlite.SQLite3Database):
        #     atoms_iter = atoms_collection.select()
        # elif isinstance(
        #     atoms_collection, ase.io.trajectory.SlicedTrajectory
        # ) or isinstance(atoms_collection, ase.io.trajectory.TrajectoryReader):
        #     atoms_iter = atoms_collection
        else:
            raise NotImplementedError

        if ref_positions_list is None:
            ref_positions_list = [
                atoms.get_positions(wrap=True) for atoms in atoms_iter
            ]

        if weights_list is None:
            weights_list = [
                [1.0] * len(ref_positions) for ref_positions in ref_positions_list
            ]

        elif isinstance(weights_list, list):
            temp_weights_list = []
            for i, weights in enumerate(weights_list):
                if isinstance(weights, list):
                    assert len(weights) == len(ref_positions_list[i])
                    temp_weights_list.append(weights)
                elif isinstance(weights, float):
                    temp_weights_list.append([weights] * len(ref_positions_list[i]))
                else:
                    raise ValueError
            weights_list = temp_weights_list

        else:
            raise ValueError

        assert len(ref_positions_list) == len(atoms_iter)
        # list for all data
        data_list = []
        for idx, (atoms, ref_positions, weights) in tqdm(
            enumerate(zip(atoms_iter, ref_positions_list, weights_list)),
            desc="converting ASE atoms collection to Data objects",
            total=len(atoms_collection),
            unit=" systems",
            disable=disable_tqdm,
        ):
            # # check if atoms is an ASE Atoms object this for the ase.db case
            # if not isinstance(atoms, ase.atoms.Atoms):
            #     atoms = atoms.toatoms()
            data = self.convert(
                atoms, idx, ref_positions=ref_positions, weights=weights,
            )
            data_list.append(data)

        return data_list
