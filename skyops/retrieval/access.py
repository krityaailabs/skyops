"""Lectures 9.3, 9.4 and 9.9: document-level access control, enforced at query time, by code.

Who is asking is a Principal: a subject, the tenant they work for, and their roles. It comes from the
caller's identity (a verified token, then the directory), never from the model's arguments and never from
the prompt. A retriever turns it into a filter that runs before ranking, on every query.

The leak patterns this module is tested against:
    missing filter      the index holds restricted documents and nobody filters: an agent's question about
                        compensation returns the duty managers' authority limits (MGR-007)
    filter in the prompt "only use documents for agents" in a system prompt: the restricted text is already
                        in the context window, and an injected or careless instruction can repeat it
    post-filter at k    filter AFTER taking the top k: no leak, but narrow roles get fewer results or none
    shared index        one index for every tenant with a tenant field nobody is forced to set; and shared
                        corpus statistics (document frequencies) that tell one tenant about another's documents
"""
from __future__ import annotations

from dataclasses import dataclass

from .docs import DocMeta
from skyops.model.errors import LabNotDone

ROLES = ("agent", "duty_manager", "ground_staff", "crew_control", "security")


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant: str
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = set(self.roles) - set(ROLES)
        if unknown:
            raise ValueError(f"unknown roles {sorted(unknown)}")


def can_read(p: Principal, meta: DocMeta) -> bool:
    """Lecture 9.3: may this principal read this document? Both must hold:

    - same tenant: p.tenant == meta.tenant (a YY Connect user never reads a ZZ procedure, and the reverse);
    - at least one shared role: set(p.roles) & set(meta.roles) is not empty.
    A principal with no roles reads nothing. Classification is not checked separately: restricted documents
    are restricted by their role lists (MGR-007 lists only duty_manager). Retriever.search calls this for
    every chunk, before ranking, on every search: every search check from 9.3 on needs it.
    """
    raise LabNotDone("Lecture 9.3: implement can_read(). Hints: labs/s09/README.md, lecture 9.3.")


# The directory: who works where, in which role. In production this is your identity provider's groups,
# looked up from the verified token's subject. Synthetic people, synthetic subjects.
DIRECTORY: dict[str, Principal] = {
    "agent-anna": Principal("agent-anna", "zz", ("agent",)),
    "dm-deepa": Principal("dm-deepa", "zz", ("duty_manager",)),
    "ramp-raj": Principal("ramp-raj", "zz", ("ground_staff",)),
    "sec-sam": Principal("sec-sam", "zz", ("security",)),
    "crew-chen": Principal("crew-chen", "zz", ("crew_control",)),
    "yy-agent-yusuf": Principal("yy-agent-yusuf", "yy", ("agent",)),
    "yy-dm-yara": Principal("yy-dm-yara", "yy", ("duty_manager",)),
}
