import numpy as np
from struct import unpack
from hashlib import sha1
from joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem import rdmolops, rdCIPLabeler, rdMHFPFingerprint

class MAPCalculator():

    def __init__(self, max_radius:int=2, n_permutations:int=2048, mapping:bool=False, n_cores:int=2, seed:int=42):
        """
        Initializes an instance of the MAPCalculator class with specified parameters.

        Parameters:
        ----------
        radius: The maximum radius of the circular substructures to extract. Default is 2.
        n_permutations: The number of permutations (= bits) used in the minhashing procedure. Default is 2048.
        mapping: A boolean value indicating whether to add mapping of hashes to their respective shingles. Default is False.
        n_cores: The number of CPU cores to use for parallel processing. Default is 2.
        seed: The seed value for random number generation. Default is 42.

        Example Usage:
        ----------
        ```python
        map4 = MAPCalculator(radius=2, n_permutations=2048, mapping=False, n_cores=8, seed=42)
        ```
        """
        # Initialize parameters
        self.max_radius = max_radius
        self.n_permutations = n_permutations
        self.encoder = rdMHFPFingerprint.MHFPEncoder(self.n_permutations)
        self.mapping = mapping
        self.n_cores = n_cores
        self.seed = seed

        # Initialize random permutations
        rand = np.random.RandomState(self.seed)

        # Define upper bounds
        self.prime = (1 << 61) - 1
        self.max_hash = 2**32 - 1

        # Generate random permutations
        self.permutations_a = np.zeros([n_permutations], dtype=np.uint32)
        self.permutations_b = np.zeros([n_permutations], dtype=np.uint32)

        
        for i in range(n_permutations):
            a = rand.randint(1, self.max_hash, dtype=np.uint32)
            b = rand.randint(0, self.max_hash, dtype=np.uint32)

            while a in self.permutations_a:
                a = rand.randint(1, self.max_hash, dtype=np.uint32)

            while b in self.permutations_b:
                b = rand.randint(0, self.max_hash, dtype=np.uint32)

            self.permutations_a[i] = a
            self.permutations_b[i] = b

        self.permutations_a = self.permutations_a.reshape((n_permutations, 1))
        self.permutations_b = self.permutations_b.reshape((n_permutations, 1))

        #Initialize parallel processing
        self.parallel = Parallel(n_jobs=self.n_cores, backend='threading')
    
    def get_substructures(self, mol:int, max_radius:int=None) -> np.array:
        """
        Retrieves all substructures around each atom within a specified maximum radius. Substructures where the central atom is chiral are marked according to their CIP descriptor.

        Parameters:
        ----------
        mol (RDKit Mol): The molecular structure from which substructures will be extracted.
        max_radius: The maximum radius for the substructures extraction.

        Returns:
        ----------
        np.array: An array (n_atoms x max_radius) containing substructure information for each atom within the maximum radius.

        Example Usage:
        ----------
        ```python
        # Assuming 'get_substructures' is a method of an instance 'map4' of a class called 'MAPCalculator'
        mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
        max_radius = 2
        substructures = map4.get_substructures(mol, max_radius)
        print(substructures)
        ```
        """

        if max_radius is None:
            max_radius = self.max_radius

        rdCIPLabeler.AssignCIPLabels(mol)
        chiral_centers = dict(Chem.FindMolChiralCenters(mol))
        substructures = np.empty((mol.GetNumAtoms(), max_radius), dtype='object')
        
        for atom in range(mol.GetNumAtoms()):
            for radius in range(1, max_radius+1):
                substructures[atom, radius-1] = self.get_atom_env(mol, radius, atom)
        
        for idx, label in chiral_centers.items():
            substructures[idx, max_radius-1] = f'${label}$' + substructures[idx, max_radius-1][1:]

        return substructures
    

    def get_shingles(self, mol:int, max_radius:int=None) -> list:
        """
        Generates unique shingles based on substructure information and interatomic distances.

        Parameters:
        ----------
        mol (RDKit Mol): The molecular structure from which shingles will be generated.
        max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.

        Returns:
        ----------
        shingles: A list of unique shingles generated based on substructure information and interatomic distances.

        Example Usage:
        ----------
        ```python
        # Assuming 'get_shingles' is a method of an instance 'map4' of a class called 'MAPCalculator'
        mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
        max_radius = 2
        result = map4.get_shingles(mol, max_radius)
        print(result)
        ```
        """

        if max_radius is None:
            max_radius = self.max_radius

        distance_matrix = rdmolops.GetDistanceMatrix(mol)
        substructures = self.get_substructures(mol, max_radius)
        n_atoms, max_radius = substructures.shape
        shingles = set()
        for r in range(max_radius):
            for i in range(n_atoms):
                for j in range(n_atoms):
                    shingle = f'{substructures[i, r]}|{int(distance_matrix[i, j])}|{substructures[j, r]}'
                    shingles.add(shingle.encode('utf-8'))

        return list(shingles)
    

    def get_fingerprint(self, mol:int, max_radius:int=None) -> np.array:
        """
        Generates a fingerprint by minhashing unique shingles extracted from a molecular structure.

        Parameters:
        ----------
        mol (RDKit Mol): The molecular structure from which the fingerprint will be generated.
        max_radius (int): The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.

        Returns:
        ----------
        fingerprint: An array representing the fingerprint generated by minhashing the unique shingles.

        Example Usage:
        ----------
        ```python
        # Assuming 'get_fingerprint' is a method of an instance 'map4' of a class called 'MAPCalculator'
        mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
        max_radius = 2
        fingerprint = map4.get_fingerprint(mol, max_radius)
        print(fingerprint)
        ```
        """

        if max_radius is None:
            max_radius = self.max_radius

        shingles = self.get_shingles(mol, max_radius)
        fingerprint = self.encoder.FromStringArray(shingles)
        return np.array(fingerprint, dtype=np.uint32)
    

    def get_fingerprints(self, mols:list, max_radius:int=None) -> list:
        """
        Generates a list of fingerprints from a list of molecular structures. The function uses parallel processing to speed up the generation of fingerprints.

        Parameters:
        ----------
        mols: A list of RDKit Mol objects representing the molecular structures.
        max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.

        Returns:
        ----------
        fingerprints: A list of arrays representing the fingerprints generated for each molecular structure in the input list.

        Example Usage:
        ----------
        ```python
        # Assuming 'get_fingerprints' is a method of an instance 'map4' of a class called 'MAPCalculator'
        mols = [Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O'), Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@H]1N2C(=O)C3=CC=CC=C3C2=O')]
        max_radius = 2
        fingerprints = map4.get_fingerprints(mols, max_radius)
        print(fingerprints)
        ```
        """

        if max_radius is None:
            max_radius = self.max_radius

        fingerprints = self.parallel(delayed(self.get_fingerprint)(mol, max_radius) for mol in mols)

        return fingerprints
    

    def get_hash_dict(self, mol, max_radius:int=None) -> dict:
        """
        Generates a dictionary linking the generated shingles to their unique hashes.
        Warning: This function should only be used when mapping the shingles to their corresponding substructures is required. Else use the 'get_fingerprint' function.

        Parameters:
        ----------
        mol (RDKit Mol): The molecular structure from which the dictionary will be generated.
        max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.

        Returns:
        ----------
        hash_dict: A dictionary containing hashes as keys and their corresponding shingles as values.

        Example Usage:
        ----------
        ```python
        # Assuming 'get_hash_dict' is a method of an instance 'map4' of a class called 'MAPCalculator'
        mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
        max_radius = 2
        hash_dict = map4.get_hash_dict(mol, max_radius)
        print(hash_dict)
        ```
        """

        if max_radius is None:
            max_radius = self.max_radius

        shingles = self.get_shingles(mol, max_radius)
        hash_dict = {}
        for shingle in shingles:
            hash_dict[unpack("<I", sha1(shingle).digest()[:4])[0]] = shingle
        return hash_dict
    

    def get_fingerprint_with_mapping(self, mol, max_radius:int=None) -> tuple:
        """
        Generates a fingerprint along with a dictionary that maps the fingerprint values to their corresponding shingles.
        Warning: This function should only be used when mapping the shingles to their corresponding substructures is required. Else use the 'get_fingerprint' function.

        Parameters:
        ----------
        mol (RDKit Mol): The molecular structure from which the fingerprint and mapping will be generated.
        max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.

        Returns:
        ----------
        fingerprint, hash_map: A tuple containing an array representing the fingerprint values and a dictionary representing the mapping of those values to the corresponding shingles.

        Example Usage:
        ----------
        ```python
        # Assuming 'get_fingerprint_with_mapping' is a method of an instance 'map4' of a class called 'MAPCalculator'
        mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
        max_radius = 2
        fingerprint, hash_map = map4.get_fingerprint_with_mapping(mol, max_radius)
        print(fingerprint)
        print(hash_map)
        ```
        """

        if max_radius is None:
            max_radius = self.max_radius

        hash_dict = self.get_hash_dict(mol, max_radius)
        hash_values = np.full((self.n_permutations, 1), self.max_hash, dtype=np.uint32)

        hash_map_full = {}
        for hash, shingle in hash_dict.items():
            hashes = np.remainder(np.remainder(self.permutations_a * hash + self.permutations_b, self.prime), self.max_hash).astype(np.uint32)
            hash_values = np.minimum(hash_values, hashes)
            hash_intersection = np.intersect1d(hash_values, hashes)
            
            for h in hash_intersection:
                hash_map_full[h] = shingle

        hash_values = hash_values.reshape((1, self.n_permutations))[0]

        hash_map = {}
        for key in hash_values:
            hash_map[key] = hash_map_full[key]

        return hash_values, hash_map
    

    def get_fingerprints_with_mapping(self, mols:list, max_radius:int=None) -> tuple:
        """
        Generates a list of fingerprints and a dictionary linking all fingerprint values to their original shingles.
        Warning: This function should only be used when mapping the shingles to their corresponding substructures is required. Else use the 'get_fingerprints' function.

        Parameters:
        ----------
        mols: A list of RDKit Mol objects representing the molecular structures.
        max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.

        Returns:
        ----------
        fingerprints, hash_map: A tuple containing a list of arrays representing the fingerprints and a combined dictionary representing the mappings all values to the corresponding shingles.

        Example Usage:
        ----------
        ```python
        # Assuming 'get_fingerprints_with_mapping' is a method of an instance 'map4' of a class called 'MAPCalculator'
        mols = [Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O'), Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@H]1N2C(=O)C3=CC=CC=C3C2=O')]
        max_radius = 2
        fingerprints, hash_map = processor.get_fingerprints_with_mapping(mols, max_radius)
        print(fingerprints)
        print(hash_map)
        ```
        """

        if max_radius is None:
            max_radius = self.max_radius

        fingerprints_dicts = self.parallel(delayed(self.get_fingerprint_with_mapping)(mol, max_radius) for mol in mols)

        fingerprints = [item[0] for item in fingerprints_dicts]
        all_dicts = [item[1] for item in fingerprints_dicts]
        combined_dict = self.combine_dicts(all_dicts)
        
        return fingerprints, combined_dict


    def encode(self, mol, max_radius:int=None, mapping:bool=None) -> tuple or np.array:
        """
        Encodes the molecular structure into a fingerprint. If mapping=True, the function returns the fingerprint and a dictionary mapping the hashes to their corresponding shingles of origin.

        Parameters:
        ----------
        mol (RDKit Mol): The molecular structure to be encoded into a fingerprint.
        max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.
        mapping: A boolean flag indicating whether to include mapping in the encoding process.

        Returns:
        ----------
        tuple or np.array: If 'mapping' is True, returns a tuple containing an array representing the fingerprint values and a dictionary representing the mapping of those values to the corresponding shingles. If 'mapping' is False, returns an array representing the generated fingerprint.

        Example Usage:
        ----------
        ```python
        # Assuming 'encode' is a method of an instance 'map4' of a class called 'MAPCalculator'
        mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
        max_radius = 2

        # For mapping = False
        mapping = False
        fingerprint = map4.encode(mol, max_radius, mapping)
        print(fingerprint)

        # For mapping = True
        mapping = True
        fingerprint, hash_map = map4.encode(mol, max_radius, mapping)
        print(fingerprint)
        print(hash_map)
        ```
        """

        if max_radius is None:
            max_radius = self.max_radius

        if mapping is None:
            mapping = self.mapping

        if mapping:
            return self.get_fingerprint_with_mapping(mol, max_radius)
        else:
            return self.get_fingerprint(mol, max_radius)
    

    def encode_many(self, mols:list, max_radius:int=None, mapping:bool=None):
        """
        Encodes multiple molecular structures into fingerprints with or without mapping. If mapping=True, the function returns the fingerprints and a dictionary mapping the hashes to their corresponding shingles of origin.

        Parameters:
        ----------
        mols: A list of RDKit Mol objects representing the molecular structures to be encoded.
        max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.
        mapping: A boolean flag indicating whether to include mapping in the encoding process.

        Returns:
        ----------
        tuple or list: If 'mapping' is True, returns a tuple containing a list of arrays representing the fingerprints and a combined dictionary representing the mappings of all values to the corresponding shingles. If 'mapping' is False, returns a list of arrays representing the generated fingerprints.

        Example Usage:
        ----------
        ```python
        # Assuming 'encode_many' is a method of an instance 'map4' of a class called 'MAPCalculator'
        mols = [Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O'), Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@H]1N2C(=O)C3=CC=CC=C3C2=O')]
        max_radius = 2

        # For mapping = False
        mapping = False
        fingerprints = map4.encode_many(mols, max_radius, mapping)
        print(fingerprints)

        # For mapping = True
        mapping = True
        fingerprints, hash_map = map4.encode_many(mols, max_radius, mapping)
        print(fingerprints)
        print(hash_map)
        ```
        """

        if max_radius is None:
            max_radius = self.max_radius

        if mapping is None:
            mapping = self.mapping

        if mapping:
            return self.get_fingerprints_with_mapping(mols, max_radius)
        else:
            return self.get_fingerprints(mols, max_radius)


    @staticmethod
    def get_atom_env(mol, radius:int, atom:int) -> str:
        """
        Extracts the local chemical environment around a specified atom within a molecular structure.

        Parameters:
        ----------
        mol (RDKit Mol): The molecular structure from which the atom environment will be extracted.
        radius: The radius of the desired atom environment. Defines the number of bonds away from the central atom to be considered in the local environment.
        atom: The index of the target atom in the molecular structure. The function will extract the environment around this specific atom.

        Returns:
        ----------
        smiles: The SMILES representation of the identified substructure.

        Example Usage:
        ----------
        ```python
        # Assuming 'get_atom_env' is a method of an instance 'map4' of a class called 'MAPCalculator'
        mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
        atom_index = 5
        radius = 2
        substructure = map4.get_atom_env(mol, radius, atom_index)
        print(substructure)
        ```
        """
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom)
        amap = {}
        mol = Chem.PathToSubmol(mol, env, atomMap=amap)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=False, rootedAtAtom=amap[atom]).replace('[CH]', 'C') # Manual correction to ensure comparability with nonchiral SMILES.
        return smiles


    @staticmethod
    def combine_dicts(dict_list:list) -> dict:
        """
        Combines multiple dictionaries into a single dictionary.

        Parameters:
        dict_list: A list of dictionaries to be combined.

        Returns:
        dict: A dictionary containing the combined key-value pairs from all input dictionaries.

        Example Usage:
        ```python
        dict_list = [{'a': 1, 'b': 2}, {'c': 3, 'd': 4}, {'e': 5, 'f': 6}]
        result = combine_dicts(dict_list)
        print(result)
        ```
        """
        combined_dict = {}
        for d in dict_list:
            combined_dict.update(d)
        return combined_dict