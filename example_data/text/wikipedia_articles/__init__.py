from pathlib import Path


FILE_PREFIX: str = "wikipedia-"
FILE_SUFFIX: str = ".txt"


def load_text(article_name: str) -> str:

    if (len(article_name) < len(FILE_PREFIX)) or (article_name[len(FILE_PREFIX)] != FILE_PREFIX):
        article_name = FILE_PREFIX + article_name

    if (len(article_name) < len(FILE_SUFFIX)) or (article_name[-len(FILE_SUFFIX):] != FILE_SUFFIX):
        article_name = article_name + FILE_SUFFIX

    path = (Path(__file__).parent.absolute())/article_name

    return path.read_text()
