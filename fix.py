content = open('tests/test_system_integrity.py', encoding='utf-8').read()
content = content.replace('"config.BASE_ASSET_LONG"', 'config.BASE_ASSET_LONG')
content = content.replace("'config.BASE_ASSET_LONG'", 'config.BASE_ASSET_LONG')
open('tests/test_system_integrity.py', 'w', encoding='utf-8').write(content)
