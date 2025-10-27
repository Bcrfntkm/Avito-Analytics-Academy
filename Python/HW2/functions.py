import csv
import argparse
from typing import List, Dict


def get_heirarchy(df: Dict) -> Dict:

    """
    Группирует информацию по департаментам, возвращает словарь, в котором
    ключи - названия департаментов, значения - списки
    с информацией по каждому сотруднику.
    """
    
    heirarchy = dict()
    for name, teams in df.items():
        teams = list(map(lambda row: row[1], teams))
        heirarchy[name] = list(set(teams))
    
    return heirarchy


def get_statistics(df: Dict) -> List[Dict]:

    """
    Формирует статистику по департаментам, а именно: название, количество
    сотрудников, минимальная, максимальная и средняя зарплаты
    """

    return_value = []
    for name, teams in df.items():
        salary = list(map(lambda row: float(row[-1]), teams))
        return_value.append({
            'name': name,
            'staff': len(teams),
            'min_salary': min(salary),
            'max_salary': max(salary),
            'average_salary': sum(salary) / len(teams)
         })
    
    return return_value


def write_statistics_into_csv(statistics: List[Dict], headers: List[str], output_file: str) -> None:

    """
    Формирует файл со статистикой для 3 сценария. Получает на вход статистику,
    заголовки и название выходного файла
    """

    with open(output_file, 'w', encoding='utf-8') as csv_file:
        csv_file.write(', '.join(headers) + '\n')
        
        for item in statistics:
            row = []
            for header in headers:
                value = str(item.get(header, ''))
                if ',' in value or '"' in value:
                    value = f'"{value.replace(chr(34), chr(34)+chr(34))}"'
                if header != 'name':
                    value = str(round(float(value), 2))
                row.append(value)
            csv_file.write(', '.join(row) + '\n')
    print(f"Отчет успешно записан в {output_file}")
    return


def main():

    """
    Основная функция программы. Считывает в аргументах
    командной строки через флаги -f или --file
    название файла со статистикой по работникам. Ожидает
    числа из диапазона 1, 2, 3, обрабатывает соответствующий
    сценарий в соответствии с логикой, описанной
    в задании, то есть выводит в стандартный поток вывода либо текст,
    либо формирует файл со статистикой с названием 'final_report.csv'.
    """

    parser = argparse.ArgumentParser(description="Чтение файла")
    parser.add_argument('-f', '--file', help="Имя файла", required=True)
    args = parser.parse_args()
    departments = dict()

    try:
        with open(args.file, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                row = row[0].split(';')
                if row[1] != 'Департамент':
                    if row[1] not in departments:
                        departments[row[1]] = [row[:1] + row[2:]]
                    else:
                        departments[row[1]].append(row[:1] + row[2:])
    except FileNotFoundError:
        print(f'Файл {args.file} не найден')
    except Exception as e:
        print(f'Ошибка: {e}')

    scenario = 0

    while scenario not in {1, 2, 3}:
        scenario = input('Введите число от 1 до 3:\n' \
        '1. Вывести в понятном виде иерархию команд, т.е. департамент и все команды, которые входят в него\n' \
        '2. Вывести сводный отчёт по департаментам: название, численность, "вилка" зарплат в виде мин – макс, среднюю зарплату\n' \
        '3. Сохранить сводный отчёт из предыдущего пункта в виде csv-файла.\n')
        try: 
            scenario = int(scenario)
        except Exception:
            print('Попробуйте ещё раз.')
            continue

        if scenario == 1:
            print("Департамент: Команды")
            heirarchy = get_heirarchy(departments)
            for department, teams in heirarchy.items():
                print(department, ": ", end='')
                for team in teams:
                    print(team, end=', ')
                print()
        elif scenario == 2:
            print("Department:")
            stats = get_statistics(departments)
            print(stats)
            print("Департамент - Кол-во сотрудников - Мин. - Макс. - Средняя зарплата")
            for doc in stats:
                name, staff, min_, max_, average = doc.values()
                print(f'{name}: {staff} {min_:.2f} - {max_:.2f} {average:.2f}')
        elif scenario == 3:
            statistics = get_statistics(departments)
            headers = list(statistics[0].keys())

            output_file = 'final_report.csv'
            write_statistics_into_csv(statistics, headers, output_file)

     
if __name__ == "__main__":
    main()
