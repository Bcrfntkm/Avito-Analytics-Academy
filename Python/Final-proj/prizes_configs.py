from json_dict_processing import create_processor
from datetime import datetime

CONFIG_PRIZE = {
    "prize_amount": ["prizeAmount"],
    "prize_amount_adjusted": ["prizeAmountAdjusted"],
    "award_year": (["awardYear"], int),
    "category_en": ["category", "en"],
    "prize_status": ["prizeStatus"],
}


def process_year(year_string):
    try:
        return int(str(year_string)[:4])
    except (ValueError, TypeError):
        return None


prize_processor = create_processor(CONFIG_PRIZE, list_processor=False)
prizes_processor = create_processor(CONFIG_PRIZE, list_processor=True)
