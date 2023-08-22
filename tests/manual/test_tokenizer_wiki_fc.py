from example_data.text.wikipedia_articles import load_text
from text.tokenizing import tokenize, WordTextToken, EndOfSectionTextToken


print("\t".join(
    [
        token.word if isinstance(token, WordTextToken)
        else "---" if isinstance(token, EndOfSectionTextToken)
        else ""
        for token in tokenize(load_text("frances-cleveland"))
    ]
))
