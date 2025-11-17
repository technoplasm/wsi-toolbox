
from pydantic import BaseModel
from pydantic_autocli import AutoCLI

from .seed import fix_global_seed, get_global_seed


class BaseMLArgs(BaseModel):
    seed: int = get_global_seed()

class BaseMLCLI(AutoCLI):
    class CommonArgs(BaseMLArgs):
        pass

    def _pre_common(self, a:BaseMLArgs):
        fix_global_seed(a.seed)
        super()._pre_common(a)
