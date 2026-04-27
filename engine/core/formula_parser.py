"""
engine/formula_parser.py — String formül → Node AST parser.

Format:
  Leaves : "Pclose", "0.05", "20"
  Inner  : "Op(arg1, arg2, ...)"   ör: "Rank(Mul(Pclose, Vlot), 20)"

LLM çıktılarını replay buffer'a aktarmak için kullanılır.
"""
from typing import List, Tuple

from .alpha_cfg import AlphaCFG, Node


class ParseError(ValueError):
    pass


# LLM'lerin ürettiği yaygın alternatif isimler → bizim operatör adlarımız
_ALIASES: dict = {
    # Cross-section
    "cs_rank": "CSRank", "csrank": "CSRank", "rank_cs": "CSRank",
    # Time-series rolling
    "ts_mean": "Mean", "ts_avg": "Mean",
    "ts_sum": "Sum",
    "ts_std": "Std", "ts_stddev": "Std",
    "ts_var": "Var", "ts_variance": "Var",
    "ts_max": "Max",
    "ts_min": "Min",
    "ts_rank": "Rank",
    "ts_skew": "Skew", "ts_skewness": "Skew",
    "ts_kurt": "Kurt", "ts_kurtosis": "Kurt",
    "ts_med": "Med", "ts_median": "Med",
    "ts_mad": "Mad",
    "ts_delta": "Delta", "ts_diff": "Delta",
    "ts_decay": "WMA", "ts_decay_linear": "WMA", "decay_linear": "WMA",
    "ts_ema": "EMA",
    "ts_wma": "WMA",
    "ts_ref": "Ref", "delay": "Ref", "ts_delay": "Ref",
    "ts_corr": "Corr",
    "ts_cov": "Cov",
    # Unary (lowercase LLM varyantları)
    "log": "Log", "ln": "Log",
    "abs": "Abs", "absolute": "Abs",
    "sign": "Sign", "signum": "Sign",
    # Binary
    "add": "Add",
    "sub": "Sub", "subtract": "Sub",
    "mul": "Mul", "multiply": "Mul", "product": "Mul",
    "div": "Div", "divide": "Div",
    "pow": "Pow", "power": "Pow",
    "greater": "Greater", "ts_max2": "Greater",
    "less": "Less", "ts_min2": "Less",
    # Feature aliases
    "open": "Popen", "popen": "Popen",
    "high": "Phigh", "phigh": "Phigh",
    "low": "Plow", "plow": "Plow",
    "close": "Pclose", "pclose": "Pclose",
    "volume": "Vlot", "vol": "Vlot", "vlot": "Vlot",
    "vwap": "Pvwap", "pvwap": "Pvwap",
}


def _normalize(name: str) -> str:
    """Operatör/özellik adını normalize eder (alias → canonical)."""
    return _ALIASES.get(name.lower(), name)


def _build_op_kind_map(cfg: AlphaCFG) -> dict:
    """Operatör adını AST kind'ına eşler."""
    m = {}
    for op in cfg.UNARY_OPS:        m[op] = "unary"
    for op in cfg.BINARY_OPS:       m[op] = "binary"
    for op in cfg.BINARY_ASYM_OPS:  m[op] = "binary_asym"
    for op in cfg.ROLLING_OPS:      m[op] = "rolling"
    for op in cfg.PAIRED_OPS:       m[op] = "paired_rolling"
    for op in cfg.CS_OPS:           m[op] = "cs_op"
    return m


def _tokenize(s: str) -> List[Tuple[str, str]]:
    """Basit tokenizer: ident, number, ( ) , boşluk."""
    toks: List[Tuple[str, str]] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "(),":
            toks.append(("punct", c))
            i += 1
            continue
        # number (negatif dahil)
        if c.isdigit() or (c == "-" and i + 1 < n and (s[i+1].isdigit() or s[i+1] == ".")) \
           or c == ".":
            j = i + 1
            while j < n and (s[j].isdigit() or s[j] in ".eE+-"):
                # 1e-5 gibi ifadeleri yakala ama sonraki (-)'yi işaret olarak alma
                if s[j] in "+-" and j > 0 and s[j-1] not in "eE":
                    break
                j += 1
            toks.append(("num", s[i:j]))
            i = j
            continue
        # identifier
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            toks.append(("ident", s[i:j]))
            i = j
            continue
        raise ParseError(f"beklenmedik karakter: {c!r} @ {i}")
    return toks


class _Parser:
    def __init__(self, cfg: AlphaCFG, tokens: List[Tuple[str, str]]):
        self.cfg = cfg
        self.op_kind = _build_op_kind_map(cfg)
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ("eof", "")

    def _eat(self, ttype=None, tval=None):
        tok = self._peek()
        if ttype and tok[0] != ttype:
            raise ParseError(f"beklenen tür {ttype}, bulundu {tok}")
        if tval and tok[1] != tval:
            raise ParseError(f"beklenen {tval!r}, bulundu {tok}")
        self.pos += 1
        return tok

    def parse_expr(self) -> Node:
        tok = self._peek()
        if tok[0] == "num":
            self._eat()
            val = float(tok[1])
            # NUMS listesinde ise "num", değilse "constant"
            if any(abs(val - n) < 1e-9 for n in self.cfg.NUMS):
                return Node("num", str(int(val) if val == int(val) else val))
            return Node("constant", str(val))
        if tok[0] == "ident":
            raw_name = tok[1]
            name = _normalize(raw_name)
            self._eat()
            nxt = self._peek()
            # Çağrı değilse → feature / sayı-isimli sabit
            if nxt != ("punct", "("):
                if name in self.cfg.FEATURES:
                    return Node("feature", name)
                raise ParseError(f"bilinmeyen leaf: {raw_name!r} (normalize: {name!r})")
            # Fonksiyon çağrısı
            self._eat("punct", "(")
            args: List[Node] = []
            if self._peek() != ("punct", ")"):
                args.append(self.parse_expr())
                while self._peek() == ("punct", ","):
                    self._eat()
                    args.append(self.parse_expr())
            self._eat("punct", ")")
            kind = self.op_kind.get(name)
            if kind is None:
                raise ParseError(f"bilinmeyen operatör: {raw_name!r} (normalize: {name!r})")
            return Node(kind, name, args)
        raise ParseError(f"beklenmedik token: {tok}")


def parse_formula(s: str, cfg: AlphaCFG) -> Node:
    """Formül string'ini AST'ye çevirir. Sözdizimi hatasında ParseError fırlatır."""
    toks = _tokenize(s)
    if not toks:
        raise ParseError("boş formül")
    p = _Parser(cfg, toks)
    tree = p.parse_expr()
    if p.pos != len(toks):
        raise ParseError(f"fazla token: {toks[p.pos:]!r}")
    return tree


def parse_many(text: str, cfg: AlphaCFG) -> List[Tuple[str, Node, str]]:
    """
    Çok satırlı girdiyi parse eder. Her satır bir formül.
    Dönüş: [(orijinal_satir, Node, ""), ...] veya hata durumunda
           [(orijinal_satir, None, "hata mesajı"), ...]
    """
    out: List[Tuple[str, Node, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append((line, parse_formula(line, cfg), ""))
        except ParseError as e:
            out.append((line, None, str(e)))
    return out
