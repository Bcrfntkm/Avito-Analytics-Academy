def extract_nested_value(obj, keys):
    current = obj
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
            current = current[int(key)]
        else:
            return None
    return current


def process_dictionary_with_config(dictionary, config):
    result = {}

    for attribute, path_config in config.items():
        if isinstance(path_config, list):
            value = extract_nested_value(dictionary, path_config)
            result[attribute] = value
        elif isinstance(path_config, tuple) and len(path_config) == 2:
            path, processor_func = path_config
            raw_value = extract_nested_value(dictionary, path)
            if raw_value is not None:
                try:
                    result[attribute] = processor_func(raw_value)
                except Exception:
                    result[attribute] = None
            else:
                result[attribute] = None
        else:
            result[attribute] = None

    return result


def process_list_of_dicts_with_config(list_of_dicts, config):
    return [process_dictionary_with_config(d, config) for d in list_of_dicts]


def create_processor(config, list_processor=False):
    if list_processor:
        return lambda list_of_dicts: process_list_of_dicts_with_config(
            list_of_dicts, config
        )
    else:
        return lambda dictionary: process_dictionary_with_config(dictionary, config)
