def parse_difficulty(difficulty_path: str, difficulty) -> None:
    """Parses the difficulty file."""
    with open(difficulty_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    sections = content.split('\n\n')
    current_section = None
    current_content = []
    
    for section in sections:            
        lines = section.split('\n')
        for line in lines:
            if line.startswith('[') and line.endswith(']'):
                if current_section and current_content and current_content != "metadata":
                    _parse_section(current_section, '\n'.join(current_content), difficulty)
                current_section = line[1:-1]
                current_content = []
            else:
                current_content.append(line)
            
        if current_section and current_content:
            _parse_section(current_section, '\n'.join(current_content), difficulty)
            
def _parse_section(section_name: str, content: str, difficulty) -> None:
    """Parses the section of the difficulty file."""
    if section_name == "General":
        difficulty.general.parse(content)
    elif section_name == "Metadata":
        difficulty.metadata.parse(content)
    elif section_name == "Difficulty":
        difficulty.difficulty.parse(content)
    elif section_name == "TimingPoints":
        difficulty.timing_points.parse(content)
    elif section_name == "Colours":
        difficulty.colours.parse(content)
    elif section_name == "HitObjects":
        difficulty.hit_objects.parse(content, difficulty.difficulty.get_approach_time())