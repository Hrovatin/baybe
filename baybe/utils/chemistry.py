"""Chemistry tools."""

import os
import ssl
import tempfile
import urllib.request
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory

from baybe._optional.chem import (
    BaseFingerprintTransformer,
    Chem,
    ConformerGenerator,
    MolFromSmilesTransformer,
    skfp_fingerprints,
)
from baybe.parameters.enum import SubstanceEncoding
from baybe.utils.numerical import DTypeFloatNumpy

# Caching
_cachedir = os.environ.get(
    "BAYBE_CACHE_DIR", str(Path(tempfile.gettempdir()) / ".baybe_cache")
)


def _dummy_wrapper(func):
    return func


_disk_cache = _dummy_wrapper if _cachedir == "" else Memory(Path(_cachedir)).cache


def name_to_smiles(name: str) -> str:
    """Convert from chemical name to SMILES string using chemical identifier resolver.

    This script is useful to combine with ``df.apply`` from pandas, hence it does not
    throw exceptions for invalid molecules but instead returns an empty string for
    easy subsequent postprocessing of the dataframe.

    Args:
        name: Name or nickname of compound.

    Returns:
        SMILES string corresponding to chemical name.
    """
    name = name.replace(" ", "%20")

    try:
        url = "http://cactus.nci.nih.gov/chemical/structure/" + name + "/smiles"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(url, context=ctx) as web:
            smiles = web.read().decode("utf8")

        smiles = str(smiles)
        if "</div>" in smiles:
            return ""

        return smiles

    except Exception:
        return ""


@lru_cache(maxsize=None)
@_disk_cache
def _molecule_to_fingerprint_features(
    fingerprint_encoder: BaseFingerprintTransformer,
    molecule: str | Chem.PropertyMol.PropertyMol,
) -> np.ndarray:
    """Compute molecular fingerprint for a single SMILES string.

    Args:
        fingerprint_encoder: Instance of Fingerprint class used to
            transform smiles string to fingerprint
        molecule: Smiles string or molecule object,
            depending on what should be input into fingerprint_encoder's transform

    Returns:
        Array containing fingerprint for SMILES string.
    """
    return fingerprint_encoder.transform([molecule])


def smiles_to_fingerprint_features(
    smiles: list[str],
    encoding: SubstanceEncoding,
    prefix: str = "",
    kwargs_conformer: dict | None = None,
    kwargs_fingerprint: dict | None = None,
) -> pd.DataFrame:
    """Compute molecular fingerprints for a list of SMILES strings.

    Args:
        smiles: List of SMILES strings.
        encoding: Encoding used to transform SMILES to fingerprints.
        prefix: Name prefix for each descriptor (e.g., nBase --> <prefix>_nBase).
        kwargs_conformer: kwargs for conformer generator
        kwargs_fingerprint: kwargs for fingerprint generator

    Returns:
        Dataframe containing fingerprints for each SMILES string.
    """
    kwargs_fingerprint = kwargs_fingerprint or {}
    kwargs_conformer = kwargs_conformer or {}

    if encoding is SubstanceEncoding.MORGAN_FP:
        warnings.warn(
            f"Substance encoding '{encoding.name}' is deprecated and will be disabled "
            f"in a future version. Use '{SubstanceEncoding.ECFP.name}' "
            f"with 'fp_size' 1204 and 'radius' 4 instead.",
            DeprecationWarning,
        )
        encoding = SubstanceEncoding.ECFP
        kwargs_fingerprint.update({"fp_size": 1024, "radius": 4})

    elif encoding is SubstanceEncoding.RDKIT:
        warnings.warn(
            f"Substance encoding '{encoding.name}' is deprecated and will be disabled "
            f"in a future version. Use '{SubstanceEncoding.RDKIT2DDESCRIPTORS.name}' "
            f"instead.",
            DeprecationWarning,
        )
        encoding = SubstanceEncoding.RDKIT2DDESCRIPTORS

    fingerprint_cls = get_fingerprint_class(encoding)
    fingerprint_encoder = fingerprint_cls(**kwargs_fingerprint)

    if fingerprint_encoder.requires_conformers:
        mol_list = ConformerGenerator(**kwargs_conformer).transform(
            MolFromSmilesTransformer().transform(smiles)
        )
    else:
        mol_list = smiles

    features = np.concatenate(
        [
            _molecule_to_fingerprint_features(
                fingerprint_encoder=fingerprint_encoder, molecule=mol
            )
            for mol in mol_list
        ]
    )
    name = f"skfp{fingerprint_encoder.__class__.__name__.replace('Fingerprint', '')}_"
    col_names = [prefix + name + f for f in fingerprint_encoder.get_feature_names_out()]
    df = pd.DataFrame(features, columns=col_names, dtype=DTypeFloatNumpy)

    return df


def get_fingerprint_class(encoding: SubstanceEncoding) -> BaseFingerprintTransformer:
    """Retrieve the fingerprint class corresponding to a given encoding.

    Args:
        encoding: A substance encoding.

    Raises:
        ValueError: If no fingerprint class for the specified encoding.

    Returns:
        The fingerprint class.
    """
    # Exception case
    if encoding is SubstanceEncoding.RDKITFINGERPRINT:
        return skfp_fingerprints.RDKitFingerprint

    try:
        cls_name = next(
            name
            for name in dir(skfp_fingerprints)
            if (encoding.name + "Fingerprint").casefold() == name.casefold()
        )
    except StopIteration as e:
        raise ValueError(
            f"No fingerprint class exists for the specified encoding '{encoding.name}'."
        ) from e
    return getattr(skfp_fingerprints, cls_name)


def get_canonical_smiles(smiles: str) -> str:
    """Return the "canonical" representation of the given SMILES."""
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except Exception:
        raise ValueError(f"The SMILES '{smiles}' does not appear to be valid.")
