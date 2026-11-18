from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pwdlib.hashers.bcrypt import BcryptHasher

# Argon2 for new hashes. Bcrypt is accepted so v1 accounts can still sign in;
# their hash is upgraded to Argon2 the first time they do.
_hasher = PasswordHash((Argon2Hasher(), BcryptHasher()))

# Verifying against this when the user does not exist keeps response times even,
# so the login endpoint does not reveal which emails are registered.
_DUMMY_HASH = _hasher.hash("pathify-timing-equaliser")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, hashed: str | None) -> tuple[bool, str | None]:
    """Return (is_valid, new_hash_if_it_should_be_upgraded)."""
    if not hashed:
        _hasher.verify(password, _DUMMY_HASH)
        return False, None
    try:
        return _hasher.verify_and_update(password, hashed)
    except Exception:
        return False, None
