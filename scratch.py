import re

def extract_seconds(text):
    # Match patterns like "1 min 50 sec", "1 minute", "50 seconds", "1m50s"
    # We want to match the whole block together if possible.
    pattern = r'(?:(\d+)\s*(?:min(?:ute)?s?|m))?\s*(?:and\s*)?(?:(\d+)\s*(?:sec(?:ond)?s?|s))?'
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    times = []
    for match in matches:
        m, s = match.groups()
        if m or s:
            total = 0
            if m: total += int(m) * 60
            if s: total += int(s)
            times.append(total)
    
    # Also look for format like "1:50" or "01:05"
    pattern2 = r'(\d{1,2}):(\d{2})'
    for match in re.finditer(pattern2, text):
        m, s = match.groups()
        times.append(int(m) * 60 + int(s))
        
    return list(set(times))

print(extract_seconds("tell me about 15 sec"))
print(extract_seconds("what happens at 1 min 50 sec"))
print(extract_seconds("show me 30s"))
print(extract_seconds("tell me about 1:50 and 2:10"))
