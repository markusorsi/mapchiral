import numpy as np
from struct import unpack
from hashlib import sha1
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import rdmolops, rdCIPLabeler, rdMHFPFingerprint

def get_minhash_parameters(n_permutations:int=2048, seed:int=42): 
    """
    Generates random permutations and parameters to be used in the minhashing procedure.

    Parameters:
    ----------
    n_permutations: The number of permutations (= bits of vector) to be used in the minhashing process.
    seed: The seed value for random number generation. Default is 42.

    Returns:
    ----------
    permutations_a, permutations_b, prime, max_hash: The random permutations and parameters to be used in the minhashing procedure.

    Example Usage:
    ----------
    ```python
    permutations_a, permutations_b, prime, max_hash = get_minhash_parameters(n_permutations=2048, seed=42)
    print(permutations_a)
    print(permutations_b)
    print(prime)
    print(max_hash)
    """
    rand = np.random.RandomState(seed)

    prime = (1 << 61) - 1
    max_hash = 2**32 - 1

    permutations_a = np.zeros([n_permutations], dtype=np.uint32)
    permutations_b = np.zeros([n_permutations], dtype=np.uint32)


    for i in range(n_permutations):
        a = rand.randint(1, max_hash, dtype=np.uint32)
        b = rand.randint(0, max_hash, dtype=np.uint32)

        while a in permutations_a:
            a = rand.randint(1, max_hash, dtype=np.uint32)

        while b in permutations_b:
            b = rand.randint(0, max_hash, dtype=np.uint32)

        permutations_a[i] = a
        permutations_b[i] = b

    permutations_a = permutations_a.reshape((n_permutations, 1))
    permutations_b = permutations_b.reshape((n_permutations, 1))

    return permutations_a, permutations_b, prime, max_hash


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
    mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
    substructure = get_atom_env(mol, radius=2, atom_index=1)
    print(substructure)
    ```
    """
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom)
    amap = {}
    mol = Chem.PathToSubmol(mol, env, atomMap=amap)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True, rootedAtAtom=amap[atom]).replace('[C@H]', 'C').replace('[C@@H]', 'C') # Manual correction to preserve E/Z isomerism but remove chirality.
    return smiles


def get_substructures(mol, max_radius:int=2) -> np.array:
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
    mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
    substructures = get_substructures(mol, max_radius=2)
    print(substructures)
    ```
    """
    rdCIPLabeler.AssignCIPLabels(mol)
    chiral_centers = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    substructures = np.empty((mol.GetNumAtoms(), max_radius), dtype='object')
    
    for atom in range(mol.GetNumAtoms()):
        for radius in range(1, max_radius+1):
            substructures[atom, radius-1] = get_atom_env(mol, radius, atom)
    
    for idx, label in chiral_centers.items():
        substructures[idx, max_radius-1] = f'${label}$' + substructures[idx, max_radius-1][1:]

    return substructures


def get_shingles(mol, max_radius:int=2) -> list:
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
    mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
    result = get_shingles(mol, max_radius=2)
    print(result)
    ```
    """
    distance_matrix = rdmolops.GetDistanceMatrix(mol)
    substructures = get_substructures(mol, max_radius)
    n_atoms, max_radius = substructures.shape
    shingles = set()
    for r in range(max_radius):
        for i in range(n_atoms):
            for j in range(n_atoms):
                shingle = f'{substructures[i, r]}|{int(distance_matrix[i, j])}|{substructures[j, r]}'
                shingles.add(shingle.encode('utf-8'))

    return list(shingles)


def get_hash_dict(mol, max_radius:int=2) -> dict:
    """
    Generates a dictionary linking the generated shingles to their unique hashes.

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
    mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
    hash_dict = get_hash_dict(mol, max_radius=2)
    print(hash_dict)
    ```
    """
    shingles = get_shingles(mol, max_radius)
    hash_dict = {}
    for shingle in shingles:
        hash_dict[unpack("<I", sha1(shingle).digest()[:4])[0]] = shingle
    return hash_dict


def combine_dicts(dict_list:list) -> dict:
    """
    Combines multiple dictionaries into a single dictionary.

    Parameters:
    ----------
    dict_list: A list of dictionaries to be combined.

    Returns:
    ----------
    dict: A dictionary containing the combined key-value pairs from all input dictionaries.

    Example Usage:
    ----------
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


def get_fingerprint(mol, max_radius:int=2, n_permutations:int=2048) -> np.array:
    """
    Generates a fingerprint by minhashing unique shingles extracted from a molecular structure.

    Parameters:
    ----------
    mol (RDKit Mol): The molecular structure from which the fingerprint will be generated.
    max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.
    n_permutations: The number of permutations (= bits of vector) to be used in the minhashing process.

    Returns:
    ----------
    fingerprint: An array representing the fingerprint generated by minhashing the unique shingles.

    Example Usage:
    ----------
    ```python
    mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
    fingerprint = get_fingerprint(mol, max_radius=2, n_permutations=2048)
    print(fingerprint)
    ```
    """
    encoder = rdMHFPFingerprint.MHFPEncoder(n_permutations)
    shingles = get_shingles(mol, max_radius)
    fingerprint = encoder.FromStringArray(shingles)
    return np.array(fingerprint, dtype=np.uint32)


def get_fingerprints(mols:list, max_radius:int=2, n_permutations:int=2048, n_cpus:int=4) -> list:
    """
    Generates a list of fingerprints from a list of molecular structures. The function uses parallel processing to speed up the generation of fingerprints.

    Parameters:
    ----------
    mols: A list of RDKit Mol objects representing the molecular structures.
    max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.
    n_permutations: The number of permutations (= bits of vector) to be used in the minhashing process.
    n_cpus: The number of CPUs to be used for parallel processing.

    Returns:
    ----------
    fingerprints: A list of arrays representing the fingerprints generated for each molecular structure in the input list.

    Example Usage:
    ----------
    ```python
    mols = [Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O'), Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@H]1N2C(=O)C3=CC=CC=C3C2=O')]
    fingerprints = get_fingerprints(mols, max_radius=2, n_permutations=2048, n_cpus=4)
    print(fingerprints)
    ```
    """
    with Pool(processes=n_cpus) as pool:
        fingerprints = pool.starmap(get_fingerprint, [(mol, max_radius, n_permutations) for mol in mols])
    pool.close()
    pool.join()

    return fingerprints


def get_fingerprint_with_mapping(mol, max_radius:int=2, n_permutations:int=2048, seed:int=42) -> tuple:
    """
    Generates a fingerprint along with a dictionary that maps the fingerprint values to their corresponding shingles.
    Warning: This function should only be used when mapping the shingles to their corresponding substructures is required. Else use the 'get_fingerprint' function.

    Parameters:
    ----------
    mol (RDKit Mol): The molecular structure from which the fingerprint and mapping will be generated.
    max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.
    n_permutations: The number of permutations (= bits of vector) to be used in the minhashing process.
    seed: The seed to be used for generating the random permutations. Don't change unless necessary to ensure consistency.

    Returns:
    ----------
    fingerprint, hash_map: A tuple containing an array representing the fingerprint values and a dictionary representing the mapping of those values to the corresponding shingles.

    Example Usage:
    ----------
    ```python
    mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
    fingerprint, hash_map = map4.get_fingerprint_with_mapping(mol, max_radius=2, n_permutations=2048)
    print(fingerprint)
    print(hash_map)
    ```
    """
    permutations_a, permutations_b, prime, max_hash = get_minhash_parameters(n_permutations, seed)

    hash_dict = get_hash_dict(mol, max_radius)
    hash_values = np.full((n_permutations, 1), max_hash, dtype=np.uint32)

    hash_map_full = {}
    for hash, shingle in hash_dict.items():
        hashes = np.remainder(np.remainder(permutations_a * hash + permutations_b, prime), max_hash).astype(np.uint32)
        hash_values = np.minimum(hash_values, hashes)
        hash_intersection = np.intersect1d(hash_values, hashes)
        
        for h in hash_intersection:
            hash_map_full[h] = shingle # Intersection to be more memory efficient

    hash_values = hash_values.reshape((1, n_permutations))[0]

    hash_map = {}
    for key in hash_values:
        hash_map[key] = hash_map_full[key]

    return hash_values, hash_map
    

def get_fingerprints_with_mapping(mols:list, max_radius:int=2, n_permutations:int=2048, n_cpus:int=4, seed:int=42) -> tuple:
    """
    Generates a list of fingerprints and a dictionary linking all fingerprint values to their original shingles.
    Warning: This function should only be used when mapping the shingles to their corresponding substructures is required. Else use the 'get_fingerprints' function.

    Parameters:
    ----------
    mols: A list of RDKit Mol objects representing the molecular structures.
    max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.
    n_permutations: The number of permutations (= bits of vector) to be used in the minhashing process.
    n_cpus: The number of CPUs to be used for parallel processing.
    seed: The seed to be used for generating the random permutations. Don't change unless necessary to ensure consistency.

    Returns:
    ----------
    fingerprints, hash_map: A tuple containing a list of arrays representing the fingerprints and a combined dictionary representing the mappings all values to the corresponding shingles.

    Example Usage:
    ----------
    ```python
    mols = [Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O'), Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@H]1N2C(=O)C3=CC=CC=C3C2=O')]
    fingerprints, hash_map = get_fingerprints_with_mapping(mols, max_radius=2, n_permutations=2048, n_cpus=4)
    print(fingerprints)
    print(hash_map)
    ```
    """
    with Pool(processes=n_cpus) as pool:
        fingerprints_dicts = pool.starmap(get_fingerprint_with_mapping, [(mol, max_radius, n_permutations, seed) for mol in mols])
    pool.close()
    pool.join()

    fingerprints = [item[0] for item in fingerprints_dicts]
    all_dicts = [item[1] for item in fingerprints_dicts]
    combined_dict = combine_dicts(all_dicts)
    
    return fingerprints, combined_dict


def encode(mol, max_radius:int=2, n_permutations:int=2048, mapping:bool=False, seed:int=42) -> tuple or np.array:
    """
    Encodes the molecular structure into a fingerprint. If mapping=True, the function returns the fingerprint and a dictionary mapping the hashes to their corresponding shingles of origin.

    Parameters:
    ----------
    mol (RDKit Mol): The molecular structure to be encoded into a fingerprint.
    max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.
    n_permutations: The number of permutations (= bits of vector) to be used in the minhashing process.
    mapping: Boolean flag indicating whether to include mapping of hashes to the original shingles in the encoding process.
    seed: The seed to be used for generating the random permutations. Don't change unless necessary to ensure consistency.

    Returns:
    ----------
    tuple or np.array: If 'mapping' is True, returns a tuple containing an array representing the fingerprint values and a dictionary representing the mapping of those values to the corresponding shingles. If 'mapping' is False, returns an array representing the generated fingerprint.

    Example Usage:
    ----------
    ```python
    mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')

    # For mapping = False
    fingerprint = encode(mol, max_radius=2, n_permutations=2048, mapping=False)
    print(fingerprint)

    # For mapping = True
    fingerprint, hash_map = encode(mol, max_radius=2, n_permutations=2048, mapping=True)
    print(fingerprint)
    print(hash_map)
    ```
    """
    if mapping:
        return get_fingerprint_with_mapping(mol, max_radius, n_permutations, seed)
    else:
        return get_fingerprint(mol, max_radius, n_permutations)


def encode_many(mols, max_radius:int=2, n_permutations:int=2048, mapping:bool=False, n_cpus:int=4, seed:int=42) -> tuple or list:
    """
    Encodes multiple molecular structures into fingerprints with or without mapping. If mapping=True, the function returns the fingerprints and a dictionary mapping the hashes to their corresponding shingles of origin.

    Parameters:
    ----------
    mols: A list of RDKit Mol objects representing the molecular structures to be encoded.
    max_radius: The maximum radius for extracting substructures. Specifies the maximum distance from the central atom to consider in the local environment.
    n_permutations: The number of permutations (= bits of vector) to be used in the minhashing process.
    mapping: Boolean flag indicating whether to include mapping of hashes to the original shingles in the encoding process.
    n_cpus: The number of CPUs to be used for parallel processing.
    seed: The seed to be used for generating the random permutations. Don't change unless necessary to ensure consistency.

    Returns:
    ----------
    tuple or list: If 'mapping' is True, returns a tuple containing a list of arrays representing the fingerprints and a combined dictionary representing the mappings of all values to the corresponding shingles. If 'mapping' is False, returns a list of arrays representing the generated fingerprints.

    Example Usage:
    ----------
    ```python
    mols = [Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O'), Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@H]1N2C(=O)C3=CC=CC=C3C2=O')]

    # For mapping = False
    fingerprints = encode_many(mols, max_radius=2, n_permutations=2048, mapping=False, n_cpus=4)
    print(fingerprints)

    # For mapping = True
    fingerprints, hash_map = encode_many(mols, max_radius=2, n_permutations=2048, mapping=True, n_cpus=4)
    print(fingerprints)
    print(hash_map)
    ```
    """
    if mapping:
        return get_fingerprints_with_mapping(mols, max_radius, n_permutations, n_cpus, seed)
    else:
        return get_fingerprints(mols, max_radius, n_cpus, n_permutations)


def jaccard_similarity(fingerprint_1:np.array=None, fingerprint_2:np.array=None):
    """
    Returns the jaccard distance between two minhashed fingerprints. 
    The jaccard distance is defined as 1 - jaccard similarity.

    Parameters:
    ----------
    fingerprint_1 (np.array): The first fingerprint to be compared.
    fingerprint_2 (np.array): The second fingerprint to be compared.

    Returns:
    ----------
    float: The jaccard distance between the two fingerprints.

    Example Usage:
    ----------
    ```python
    molecule_1 = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
    molecule_2 = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@H]1N2C(=O)C3=CC=CC=C3C2=O')

    fingerprint_1 = encode(molecule_1, max_radius=2, n_permutations=2048, mapping=False)
    fingerprint_2 = encode(molecule_2, max_radius=2, n_permutations=2048, mapping=False)

    similarity = jaccard_similarity(fingerprint_1, fingerprint_2)
    print(similarity)
    """
    return np.count_nonzero(fingerprint_1 == fingerprint_2) / fingerprint_1.shape[0]