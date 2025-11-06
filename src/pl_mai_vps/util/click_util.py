import json

import click


class DictParamType(click.ParamType):
    name = 'dict'

    def convert(self, value, param, ctx):
        if isinstance(value, dict):
            return value

        try:
            # Try JSON first
            return json.loads(value)
        except json.JSONDecodeError:
            # Fall back to key=value parsing
            result = {}
            for pair in value.split(','):
                if '=' not in pair:
                    self.fail(f'Invalid format: {pair}. Use key=value or JSON.', param, ctx)
                key, val = pair.split('=', 1)
                result[key.strip()] = val.strip()
            return result


CLICK_JSON_DICT_TYPE = DictParamType()
