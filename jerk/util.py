
def exit_failure(alert_sentence : str) -> None : 
    margin_size = 5
    stress_size = len(alert_sentence) + margin_size * 2 + 2
    a = "#" * stress_size
    b = "#" * margin_size
    print()
    print(a)
    print(b + " " + alert_sentence + " " + b)
    print(a)
    print()
    assert False


def calculate_euclidean_distance(a : tuple[int, int], b : tuple[int, int]) -> int : 
    assert len(a) == 2 and len(b) == 2   # 二次元しか考慮していない
    assert a[0] == b[0] or a[1] == b[1]   # グリッドしか考慮していない
    if a[0] == b[0] : 
        return abs(a[1] - b[1])
    else : 
        return abs(a[0] - b[0])
    
