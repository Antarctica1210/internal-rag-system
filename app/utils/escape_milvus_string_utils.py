def escape_milvus_string(value: str) -> str:
    """
    Safely escape a string for use in a Milvus filter expression.
    Prevents parse errors caused by special characters in raw strings (e.g. item names, file titles).
    Escape rules:
        1. Backslash (\\) → double backslash (\\\\): required by Milvus expression syntax
        2. Double quote (") → escaped double quote (\\\"): prevents premature string termination
        3. Newline / carriage return / tab → space: keeps the expression on a single line
    :param value: Raw string to escape
    :return: Escaped string safe for use in a Milvus filter_expr
    """
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return s
