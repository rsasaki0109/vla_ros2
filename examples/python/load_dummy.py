from vla_zoo import load_model


def main() -> None:
    model = load_model("dummy")
    action = model.predict(image=None, instruction="test")
    print(action)


if __name__ == "__main__":
    main()
