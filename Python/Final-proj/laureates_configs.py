from json_dict_processing import create_processor
from prizes_configs import prizes_processor, process_year

CONFIG_PERSON = {
    "id": (["id"], int),
    "name": ["knownName", "en"],
    "gender": ["gender"],
    "birth_year": (["birth", "date"], process_year),
    "country_birth": ["birth", "place", "country", "en"],
    "country_now": ["birth", "place", "countryNow", "en"],
    "prizes_relevant": (["nobelPrizes"], prizes_processor),
}

CONFIG_ORG = {
    "id": (["id"], int),
    "name": ["orgName", "en"],
    "founded_year": (["founded", "date"], process_year),
    "country_founded": ["founded", "place", "country", "en"],
    "country_now": ["founded", "place", "countryNow", "en"],
    "prizes_relevant": (["nobelPrizes"], prizes_processor),
}

person_processor = create_processor(CONFIG_PERSON, list_processor=False)
org_processor = create_processor(CONFIG_ORG, list_processor=False)


def process_persons(persons_list):
    return [person_processor(person) for person in persons_list]


def process_orgs(orgs_list):
    return [org_processor(org) for org in orgs_list]
