import json
import re
from prompts import P2_EXPECTED, P2_LABELS, P3_EXPECTED


def classification(text):
    lines = text.strip().splitlines()
    matches = [re.fullmatch(r'\s*([1-5]):\s*(' + '|'.join(P2_LABELS) + r')\s*', line) for line in lines]
    valid = len(lines) == 5 and all(matches) and [int(m[1]) for m in matches] == list(range(1, 6))
    by_id = {}

    for line in lines:
        m = re.fullmatch(r'\s*([1-5]):\s*(' + '|'.join(P2_LABELS) + r')\s*', line)
        if m:
            by_id.setdefault(int(m[1]), []).append(m[2])

    correct = sum(by_id.get(i) == [expected] for i, expected in enumerate(P2_EXPECTED, 1))

    return {'p2_format_ok': valid, 'p2_correct': correct, 'p2_accuracy': correct / 5}


def reject_constant(value):
    raise ValueError(f'Nonstandard JSON constant: {value}')


def unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'Duplicate JSON key: {key}')
        out[key] = value
    return out


def decode(text):
    try:
        return True, json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_pairs)
    except (ValueError, TypeError):
        return False, None


def extraction(text):
    raw_ok, obj = decode(text.strip())
    recovered = raw_ok

    if not raw_ok:
        fence = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', text.strip())
        if fence:
            recovered, obj = decode(fence[1])

    result = {'p3_raw_json_valid': raw_ok, 'p3_json_after_unwrap': recovered,
              'p3_schema_ok': False, 'p3_field_score': 0, 'p3_missing_ingredients': '',
              'p3_unrecognized_ingredients': '', 'p3_transliteration': False}

    if not isinstance(obj, dict):
        return result

    types = {'dish': str, 'ingredients': list, 'water_ml': int, 'cook_time_min': int,
             'servings': int, 'is_vegan': bool, 'contains_meat': bool}

    schema = set(obj) == set(P3_EXPECTED) and all(type(obj.get(k)) is t for k, t in types.items())
    schema = schema and all(type(v) is str for v in obj['ingredients'])
    schema = schema and (obj['calories_kcal'] is None or type(obj['calories_kcal']) is int)

    result['p3_schema_ok'] = bool(schema)

    ingredients = obj.get('ingredients')

    groups = {'oats': ['овсян', 'овсян', 'овся', 'oat'], 'water': ['вод', 'water'],
              'salt': ['сол', 'salt'], 'sugar': ['сахар', 'sugar', 'sahar'], 'banana': ['банан', 'banana']}

    found, unknown = set(), []

    if isinstance(ingredients, list):
        for ingredient in ingredients:
            s = str(ingredient).casefold().replace('ё', 'е')
            matched = {name for name, roots in groups.items() if any(root in s for root in roots)}
            found.update(matched)

            if not matched:
                unknown.append(str(ingredient))

            if 'sahar' in s:
                result['p3_transliteration'] = True

    result['p3_missing_ingredients'] = ', '.join(sorted(set(groups) - found))
    result['p3_unrecognized_ingredients'] = '; '.join(unknown)

    score = int(isinstance(obj.get('dish'), str) and any(s in obj['dish'].casefold() for s in ['овся', 'oat']))
    score += int(isinstance(ingredients, list) and all(type(v) is str for v in ingredients) and len(found) == 5 and not unknown)

    for key in ['water_ml', 'cook_time_min', 'servings', 'is_vegan', 'contains_meat', 'calories_kcal']:
        expected = P3_EXPECTED[key]
        score += int(key in obj and type(obj[key]) is type(expected) and obj[key] == expected)

    result['p3_field_score'] = score

    return result


def repetition(text, n=4):
    words = re.findall(r'\w+', text.casefold())
    grams = [tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1))]
    return round(1 - len(set(grams)) / len(grams), 4) if grams else 0.0

