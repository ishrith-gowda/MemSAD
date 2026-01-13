import typer


def server(debug: bool = False, port: int = 8080):
    print("Server command invoked")
    print(f"Debug: {debug}, Port: {port}")


app = typer.Typer()
app.add_typer(typer.Typer(), name="server")

if __name__ == "__main__":
    app()
