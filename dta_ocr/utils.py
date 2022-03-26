def intersperse(input: str, delimiter: str) -> str:
    output: str = ""

    for i in range(len(input)):
        output += input[i]
        if i < len(input) - 1:
            output += delimiter

    return output
