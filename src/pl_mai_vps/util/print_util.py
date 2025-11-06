def print_segment(segment_name: str | list[str]):
    print()
    print("=" * 60)
    if isinstance(segment_name, str):
        print(segment_name)
    else:
        for s in segment_name:
            print(s)
    print("=" * 60)
    print()
