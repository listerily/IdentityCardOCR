########################################################
#
#    MODULE ID NUMBER CHECK
#      ID NUMBER CHECK checks if the ID Card number is
#    valid. If it's invalid, the server would tell user
#    that recognition has failed. If number check passed,
#    then our recognition succeeded.
#
########################################################

def check_id_length(n):
    # Length of id number must be 18
    if len(str(n)) != 18:
        return False
    else:
        return True


def check_x(n):
    # X should only appear at the last place
    for i in range(0, 16):
        if n[i] == 'X':
            return False
    return True


def year_is_legal(year):
    # Check brith year validity
    if 1900 < int(year) < 2100:
        return True
    else:
        return False


def is_ordinary_year(year):
    # check ordinary year
    if int(year) % 4 == 0 and int(year) % 100 != 0 or int(year) % 400 == 0:
        return False
    else:
        return True


def month_is_legal(month):
    # check month validity
    if 0 < int(month) < 13:
        return True
    else:
        return False


def days(year, month):
    # return maximum days in a month
    if is_ordinary_year(year):
        days = {
            '01': 31,
            '02': 28,
            '03': 31,
            '04': 30,
            '05': 31,
            '06': 30,
            '07': 31,
            '08': 31,
            '09': 30,
            '10': 31,
            '11': 30,
            '12': 31
        }
    else:
        days = {
            '01': 31,
            '02': 29,
            '03': 31,
            '04': 30,
            '05': 31,
            '06': 30,
            '07': 31,
            '08': 31,
            '09': 30,
            '10': 31,
            '11': 30,
            '12': 31
        }
    return days.get(month)


def date_is_legal(year, month, date):
    # check date validity
    if 0 < int(date) < days(year, month):
        return True
    else:
        return False


def check_id_data(n, debug):
    # ID Number validity checker using above functions
    var = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]

    # 'x'->'X'
    var_id = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
    n = str(n)
    sum = 0

    for i in range(0, 17):
        sum += int(n[i]) * var[i]
    sum %= 11
    if (var_id[sum]) == str(n[17]):
        # 获得出生年、月、日
        birth_year = n[6:10]
        birth_month = n[10:12]
        birth_date = n[12:14]
        if debug:
            print("身份证号规则核验通过，校验码是：", var_id[sum])
            print("出生于：", n[6:10], "年", n[10:12], "月", n[12:14], "日")

        if year_is_legal(birth_year) and month_is_legal(birth_month) and date_is_legal(birth_year, birth_month,
                                                                                       birth_date):

            return {
                'year': birth_year,
                'month': birth_month,
                'date': birth_date,
            }
        else:
            return None
    else:
        return None


def check_id_code(n, debug):
    # ID Number validity checker
    if check_id_length(n) and check_x(n):
        results = check_id_data(n, debug)
        if results is not None:
            return results
        else:
            return None
    else:
        return None


if __name__ == '__main__':
    # Code used for debugging
    code = input("请输入18位身份证号:")
    print(check_id_code(code, False))
