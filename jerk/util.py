
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
