import click
from colorama import Fore, Style, init

from app.src.app import FreelancerAnalyzer


init(autoreset=True)


@click.command()
def main() -> None:
    """
    Основная точка входа для приложения Freelancer AI.

    Обрабатывает интерактивный цикл вопросов и ответов.
    """
    analyzer = FreelancerAnalyzer()
    analyzer.load_data()

    print(Fore.GREEN + "Фрилансер AI запущен. Введите ваш вопрос (или 'выход' для завершения):")

    while True:
        question = click.prompt(
            Fore.YELLOW + "Ваш вопрос",
            prompt_suffix=": "
        )

        if question.lower() in ('выход', 'exit', 'quit'):
            break

        try:
            response = analyzer.ask_question(question)
            if response is None:
                print(Fore.RED + "Ошибка: Не удалось получить ответ от модели.")
            else:
                print(Fore.CYAN + "\nОтвет:")
                print(Fore.WHITE + response + "\n")
        except Exception as e:
            print(Fore.RED + f"Ошибка: {e}")


if __name__ == "__main__":
    main()