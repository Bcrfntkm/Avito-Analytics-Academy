import requests
import json
import re
import statistics
from collections import defaultdict
from laureates_configs import process_orgs, process_persons
import dataset_helpers
import aggregations
import attribute_manager
from stopwords import STOPWORDS


def get_laureate_data(laureate):
    if "knownName" in laureate:
        processed = process_persons([laureate])[0]
        processed["type_"] = "person"
        return processed
    elif "orgName" in laureate:
        processed = process_orgs([laureate])[0]
        processed["type_"] = "org"
        return processed
    else:
        return None


def calculate_age_at_first_prize(laureate):
    if "birth_year" not in laureate or laureate["birth_year"] is None:
        return None
    if "prizes_relevant" not in laureate or not laureate["prizes_relevant"]:
        return None

    award_years = [
        prize["award_year"]
        for prize in laureate["prizes_relevant"]
        if "award_year" in prize and prize["award_year"] is not None
    ]
    if not award_years:
        return None

    first_award_year = min(award_years)
    return first_award_year - laureate["birth_year"]


def get_decade(year):
    return (year // 10) * 10


def main():
    print("Fetching Nobel laureates data...")
    URL_LAUREATES = "https://api.nobelprize.org/2.1/laureates?limit=1200"
    laureates_response = requests.get(URL_LAUREATES)
    laureates = laureates_response.json()

    print("Processing laureates data...")
    laureates_data = []
    for laureate in laureates["laureates"]:
        processed_laureate = get_laureate_data(laureate)
        if processed_laureate is not None:
            laureates_data.append(processed_laureate)

    print(f"Total number of records: {len(laureates_data)}")

    person_laureates = [item for item in laureates_data if item["type_"] == "person"]
    print(f"Number of person laureates: {len(person_laureates)}")

    org_laureates = [item for item in laureates_data if item["type_"] == "org"]
    print(f"Number of organization laureates: {len(org_laureates)}")

    missing_values = {}
    total_records = len(laureates_data)
    all_fields = set()
    for item in laureates_data:
        all_fields.update(item.keys())

    for field in all_fields:
        missing_count = sum(
            1 for item in laureates_data if field not in item or item[field] is None
        )
        missing_values[field] = missing_count

    print("\nMissing values per field:")
    for field, count in missing_values.items():
        percentage = (count / total_records) * 100
        print(f"{field}: {count} ({percentage:.2f}%)")

    ids = [item["id"] for item in person_laureates if item["id"] is not None]
    if ids:
        min_id = min(ids)
        max_id = max(ids)
        print(f"\nMin ID: {min_id}")
        print(f"Max ID: {max_id}")

        min_id_item = next(item for item in person_laureates if item["id"] == min_id)
        max_id_item = next(item for item in person_laureates if item["id"] == max_id)

        min_id_years = [
            prize["award_year"]
            for prize in min_id_item["prizes_relevant"]
            if "award_year" in prize
        ]
        max_id_years = [
            prize["award_year"]
            for prize in max_id_item["prizes_relevant"]
            if "award_year" in prize
        ]

        print(f"Min ID award years: {min_id_years}")
        print(f"Max ID award years: {max_id_years}")

    print("\n=== 2.1. Топ-статистика по странам ===")
    country_groups = dataset_helpers.group_by_attributes(
        person_laureates, ["country_now"]
    )
    country_counts = {
        country[0]: len(items)
        for country, items in country_groups.items()
        if country[0] is not None
    }
    top_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Топ-5 стран по общему количеству лауреатов:")
    for i, (country, count) in enumerate(top_countries, 1):
        print(f"{i}. {country}: {count}")

    print("\nТоп стран по категориям:")
    prizes_with_country = []
    for laureate in person_laureates:
        country = laureate.get("country_now")
        if country is not None and "prizes_relevant" in laureate:
            for prize in laureate["prizes_relevant"]:
                if "category_en" in prize:
                    prizes_with_country.append(
                        {"country": country, "category": prize["category_en"]}
                    )

    category_groups = dataset_helpers.group_by_attributes(
        prizes_with_country, ["category"]
    )
    for category, items in category_groups.items():
        if category[0] is not None:
            country_groups_in_category = dataset_helpers.group_by_attributes(
                items, ["country"]
            )
            country_counts_in_category = {
                country[0]: len(prize_items)
                for country, prize_items in country_groups_in_category.items()
                if country[0] is not None
            }
            top_countries_in_category = sorted(
                country_counts_in_category.items(), key=lambda x: x[1], reverse=True
            )[:5]
            print(f"\nКатегория {category[0]}:")
            for i, (country, count) in enumerate(top_countries_in_category, 1):
                print(f"  {i}. {country}: {count}")

    print("\n=== 2.2. Возраст при награждении ===")
    person_laureates_with_age = attribute_manager.add_attribute(
        person_laureates, "age_at_first_prize", calculate_age_at_first_prize
    )
    person_laureates_with_age_data = dataset_helpers.filter_by_condition(
        person_laureates_with_age, lambda x: x["age_at_first_prize"] is not None
    )
    mean_age = aggregations.calculate_mean(
        person_laureates_with_age_data, "age_at_first_prize"
    )
    median_age = aggregations.calculate_median(
        person_laureates_with_age_data, "age_at_first_prize"
    )
    print(f"Средний возраст лауреатов при первом получении премии: {mean_age:.1f} лет")
    print(
        f"Медианный возраст лауреатов при первом получении премии: {median_age:.1f} лет"
    )

    print("\nСредний возраст по категориям:")
    prizes_with_age = []
    for laureate in person_laureates_with_age_data:
        for prize in laureate["prizes_relevant"]:
            if (
                "category_en" in prize
                and "award_year" in prize
                and prize["award_year"] is not None
            ):
                age_at_this_prize = prize["award_year"] - laureate["birth_year"]
                prizes_with_age.append(
                    {"category": prize["category_en"], "age": age_at_this_prize}
                )

    category_age_groups = dataset_helpers.group_by_attributes(
        prizes_with_age, ["category"]
    )
    for category, items in category_age_groups.items():
        if category[0] is not None:
            ages = [item["age"] for item in items]
            if ages:
                mean_age_in_category = statistics.mean(ages)
                median_age_in_category = statistics.median(ages)
                print(
                    f"  {category[0]}: средний {mean_age_in_category:.1f} лет, медианный {median_age_in_category:.1f} лет"
                )

    print("\n=== 2.3. Гендерное распределение ===")
    prizes_with_gender = []
    for laureate in person_laureates:
        gender = laureate.get("gender")
        if gender is not None and "prizes_relevant" in laureate:
            for prize in laureate["prizes_relevant"]:
                if "award_year" in prize and prize["award_year"] is not None:
                    prizes_with_gender.append(
                        {
                            "gender": gender,
                            "award_year": prize["award_year"],
                            "decade": get_decade(prize["award_year"]),
                        }
                    )

    decade_groups = dataset_helpers.group_by_attributes(prizes_with_gender, ["decade"])
    print("Количество женщин-лауреатов по десятилетиям:")
    for decade, items in sorted(decade_groups.items()):
        if decade[0] is not None:
            women_count = sum(1 for item in items if item["gender"] == "female")
            total_count = len(items)
            percentage = (women_count / total_count * 100) if total_count > 0 else 0
            print(
                f"  {decade[0]}-е: {women_count} женщин из {total_count} ({percentage:.1f}%)"
            )

    prizes_with_category_gender = []
    for laureate in person_laureates:
        gender = laureate.get("gender")
        if gender is not None and "prizes_relevant" in laureate:
            for prize in laureate["prizes_relevant"]:
                if (
                    "award_year" in prize
                    and prize["award_year"] is not None
                    and "category_en" in prize
                ):
                    prizes_with_category_gender.append(
                        {
                            "gender": gender,
                            "award_year": prize["award_year"],
                            "decade": get_decade(prize["award_year"]),
                            "category": prize["category_en"],
                        }
                    )

    category_groups = dataset_helpers.group_by_attributes(
        prizes_with_category_gender, ["category"]
    )
    print("\nЖенщины-лауреаты по категориям (всего за все годы):")
    for category, items in category_groups.items():
        if category[0] is not None:
            women_count = sum(1 for item in items if item["gender"] == "female")
            total_count = len(items)
            percentage = (women_count / total_count * 100) if total_count > 0 else 0
            print(
                f"  {category[0]}: {women_count} женщин из {total_count} ({percentage:.1f}%)"
            )

    recent_decades = sorted(
        set(
            item["decade"]
            for item in prizes_with_category_gender
            if item["decade"] is not None
        ),
        reverse=True,
    )[:3]
    print("\nЖенщины-лауреаты по категориям в последние 3 десятилетия:")
    for decade in recent_decades:
        print(f"\n  {decade}-е годы:")
        decade_items = [
            item for item in prizes_with_category_gender if item["decade"] == decade
        ]
        decade_category_groups = dataset_helpers.group_by_attributes(
            decade_items, ["category"]
        )

        for category, items in decade_category_groups.items():
            if category[0] is not None:
                women_count = sum(1 for item in items if item["gender"] == "female")
                total_count = len(items)
                percentage = (women_count / total_count * 100) if total_count > 0 else 0
                print(
                    f"    {category[0]}: {women_count} из {total_count} ({percentage:.1f}%)"
                )

    print("\n=== 2.5. Многократные лауреаты ===")
    multiple_prize_laureates = dataset_helpers.filter_by_condition(
        person_laureates,
        lambda x: "prizes_relevant" in x and len(x["prizes_relevant"]) > 1,
    )
    print(
        f"Количество лауреатов с несколькими премиями: {len(multiple_prize_laureates)}"
    )

    more_than_2_prizes = dataset_helpers.filter_by_condition(
        multiple_prize_laureates, lambda x: len(x["prizes_relevant"]) > 2
    )
    print(f"Количество лауреатов с более чем 2 премиями: {len(more_than_2_prizes)}")

    if more_than_2_prizes:
        print("\nЛауреаты с более чем 2 премиями:")
        for laureate in more_than_2_prizes:
            name = laureate.get("name", "Unknown")
            prize_count = len(laureate["prizes_relevant"])
            categories = [
                prize.get("category_en", "Unknown")
                for prize in laureate["prizes_relevant"]
            ]
            print(f"  {name}: {prize_count} премий, категории: {', '.join(categories)}")

    match_count = 0
    total_with_multiple = 0
    for laureate in multiple_prize_laureates:
        prizes = sorted(
            laureate["prizes_relevant"], key=lambda x: x.get("award_year", 0)
        )
        if len(prizes) >= 2:
            total_with_multiple += 1
            first_category = prizes[0].get("category_en")
            second_category = prizes[1].get("category_en")
            if (
                first_category is not None
                and second_category is not None
                and first_category == second_category
            ):
                match_count += 1

    if total_with_multiple > 0:
        match_percentage = (match_count / total_with_multiple) * 100
        print(f"\nСовпадение категории первой и второй премии:")
        print(
            f"  У {match_count} из {total_with_multiple} лауреатов с несколькими премиями совпадают категории первой и второй премии ({match_percentage:.1f}%)"
        )

    print("\n=== 2.6 Отказы от премий ===")
    declined_prizes = []
    for laureate in laureates_data:
        if "prizes_relevant" in laureate:
            for prize in laureate["prizes_relevant"]:
                if prize.get("prize_status") == "declined":
                    declined_prizes.append(
                        {
                            "laureate_type": laureate["type_"],
                            "category": prize.get("category_en", "Unknown"),
                        }
                    )

    print(f"Общее количество отказов от премий: {len(declined_prizes)}")
    declined_by_category = dataset_helpers.group_by_attributes(
        declined_prizes, ["category"]
    )
    print("\nОтказы по категориям:")
    for category, items in declined_by_category.items():
        if category[0] is not None:
            print(f"  {category[0]}: {len(items)} отказов")

    org_declines = [
        prize for prize in declined_prizes if prize["laureate_type"] == "org"
    ]
    print(f"\nОтказы среди организаций: {len(org_declines)}")

    print("\n=== Бонусное задание: Лингвистический анализ мотиваций ===")
    motivation_texts = []
    category_motivations = defaultdict(list)

    for laureate in laureates_data:
        if "prizes_relevant" in laureate:
            for prize in laureate["prizes_relevant"]:
                if (
                    "category_en" in prize
                    and "motivation" in prize
                    and prize["motivation"]
                ):
                    motivation = prize["motivation"]
                    motivation = (
                        motivation.replace("<p>", "")
                        .replace("</p>", "")
                        .replace('"', "")
                    )
                    motivation_texts.append(motivation)
                    category_motivations[prize["category_en"]].append(motivation)

    print(f"Собрано {len(motivation_texts)} текстов мотиваций")

    all_words = []
    for text in motivation_texts:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        all_words.extend(words)

    word_counts = {}
    for word in all_words:
        word_counts[word] = word_counts.get(word, 0) + 1

    top_50_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:50]
    print("\nТоп-50 самых частых слов:")
    for i, (word, count) in enumerate(top_50_words, 1):
        print(f"{i:2d}. {word}: {count}")

    words_without_stopwords = [word for word in all_words if word not in STOPWORDS]
    word_counts_filtered = {}
    for word in words_without_stopwords:
        word_counts_filtered[word] = word_counts_filtered.get(word, 0) + 1

    top_50_filtered = sorted(
        word_counts_filtered.items(), key=lambda x: x[1], reverse=True
    )[:50]
    print("\nТоп-50 слов после удаления стоп-слов:")
    for i, (word, count) in enumerate(top_50_filtered, 1):
        print(f"{i:2d}. {word}: {count}")

    print("\nТоп-10 слов по категориям (после удаления стоп-слов):")
    for category, texts in category_motivations.items():
        category_words = []
        for text in texts:
            words = re.findall(r"[a-zA-Z]+", text.lower())
            category_words.extend([word for word in words if word not in STOPWORDS])

        category_word_counts = {}
        for word in category_words:
            category_word_counts[word] = category_word_counts.get(word, 0) + 1

        top_10_category = sorted(
            category_word_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        print(f"\nКатегория {category}:")
        for i, (word, count) in enumerate(top_10_category, 1):
            print(f"  {i:2d}. {word}: {count}")


if __name__ == "__main__":
    main()
