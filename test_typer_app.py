import typer


def hello(name: str):
    print(f"hello {name}")


app = typer.Typer()
app.command()(hello)

if __name__ == "__main__":
    app()
