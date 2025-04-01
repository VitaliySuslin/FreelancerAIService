import click
from colorama import Fore, Style, init

from app.src.app import FreelancerAnalyzer


init(autoreset=True)  # Initialize colorama for colored console output


@click.command()
def main() -> None:
    """
    Main entry point for the Freelancer AI application.
    
    Handles the interactive question-answering loop.
    """
    # Initialize analyzer and load data
    analyzer = FreelancerAnalyzer()
    analyzer.load_data()
    
    # Print welcome message
    print(Fore.GREEN + "Фрилансер AI запущен. Введите ваш вопрос (или 'выход' для завершения):")
    
    # Main interaction loop
    while True:
        # Get user input
        question = click.prompt(
            Fore.YELLOW + "Ваш вопрос", 
            prompt_suffix=": "
        )
        
        # Check for exit commands
        if question.lower() in ('выход', 'exit', 'quit'):
            break
        
        try:
            # Get and display response
            response = analyzer.ask_question(question)
            print(Fore.CYAN + "\nОтвет:")
            print(Fore.WHITE + response + "\n")
        except Exception as e:
            # Handle errors gracefully
            print(Fore.RED + f"Ошибка: {e}")


if __name__ == "__main__":
    # Run the application
    main()