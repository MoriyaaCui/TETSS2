from tetss2 import TETSS2Predictor


def main():
    predictor = TETSS2Predictor()

    print("Testing TETSS2.0 sequence length check...\n")

    # 1. 正确长度：201 bp
    seq_201 = "ACGT" * 50 + "A"
    print(f"Test 1: 201 bp sequence, length = {len(seq_201)}")

    try:
        result = predictor.predict(seq_201)
        print("PASS: 201 bp sequence was accepted.")
        print(result)
    except Exception as e:
        print("FAIL: 201 bp sequence should be accepted, but got error:")
        print(e)

    print("\n" + "-" * 60 + "\n")

    # 2. 错误长度：16 bp
    seq_short = "ACGTACGTACGTACGT"
    print(f"Test 2: short sequence, length = {len(seq_short)}")

    try:
        result = predictor.predict(seq_short)
        print("FAIL: short sequence should be rejected, but got result:")
        print(result)
    except ValueError as e:
        print("PASS: short sequence was rejected.")
        print(e)

    print("\n" + "-" * 60 + "\n")

    # 3. 错误长度：202 bp
    seq_long = "A" * 202
    print(f"Test 3: long sequence, length = {len(seq_long)}")

    try:
        result = predictor.predict(seq_long)
        print("FAIL: long sequence should be rejected, but got result:")
        print(result)
    except ValueError as e:
        print("PASS: long sequence was rejected.")
        print(e)

    print("\n" + "-" * 60 + "\n")

    # 4. 调试模式：关闭长度检查
    print("Test 4: short sequence with check_length=False")

    try:
        result = predictor.predict(seq_short, check_length=False)
        print("PASS: short sequence was accepted when length check was disabled.")
        print(result)
    except Exception as e:
        print("FAIL: short sequence should be accepted when check_length=False.")
        print(e)


if __name__ == "__main__":
    main()