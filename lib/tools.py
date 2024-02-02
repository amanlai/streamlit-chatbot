# from calendar import day_name
from pytz import timezone
from dateutil import parser
from re import compile
from datetime import datetime, timedelta
from langchain.agents import tool
# from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
# import holidays
from langchain_community.tools import HumanInputRun


def get_special_dates(dt, fmt):

    eastern = timezone('US/Eastern')
    now = datetime.now(eastern)
    if dt in {'today', 'now'}:
        return now.strftime(fmt)
    elif dt == 'tomorrow':
        return (now + timedelta(days=1)).strftime(fmt)
    elif dt == 'yesterday':
        return (now - timedelta(days=1)).strftime(fmt)
    else:
        return None


def get_parsed_date(dt, fmt):

    if (day:=get_special_dates(dt, fmt)) is not None:
        return day
    else:
        date_pattern = compile(r'^(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-\d{4}$')
        if date_pattern.match(dt):
            try:
                the_date = parser.parse(dt)
                return the_date.strftime(fmt)
                # return day_name[the_date.weekday()]
            except:
                return 'invalid date, unable to parse'
        else:
            return 'invalid date format, please use format: mm-dd-YYYY'
    

@tool
def get_date(dt):
    """
    Returns a calendar date along with its weekday name.
    Input can be 'today', 'now', 'tomorrow' or 'yesterday' or a date in mm-dd-YYYY format.
    Used to get a calendar date or a weekday name to help answer questions related to calendar or date.
    """
    return get_parsed_date(dt, fmt='%m-%d-%Y')


@tool
def get_day_of_week(dt):
    """Returns the day of week of a given date string"""
    return get_parsed_date(dt, fmt='%A')


# @tool
# def compute_day_difference_in_days(date1, date2):
#     """Returns the difference between two dates in days"""
#     diff = datetime.strptime(date2, '%m-%d-%Y') - datetime.strptime(date1, '%m-%d-%Y')
#     return str(diff.days)


@tool
def get_delta_days_from_date(dt, delta):
    """Returns the date delta days from date dt"""
    eastern = timezone('US/Eastern')
    if dt in ('today', 'now'):
        date1 = datetime.now(eastern)
    elif dt == 'yesterday':
        date1 = datetime.now(eastern) - timedelta(days=1)
    elif dt == 'tomorrow':
        date1 = datetime.now(eastern) + timedelta(days=1)
    else:
        try:
            date1 = datetime.strptime(dt, '%m-%d-%Y')
        except ValueError:
            return 'invalid date format, please use format: mm-dd-YYYY'
    new_date = date1 + timedelta(days=int(delta))
    return new_date.strftime('%m-%d-%Y')



# us_holidays = holidays.country_holidays('US', subdiv='DC', years=date.today().year)
# us_holidays = {name: dt for dt, name in us_holidays.items()}



def get_tools():
    tools = [
        get_day_of_week,
        get_date, 
        # compute_day_difference_in_days, 
        get_delta_days_from_date, 
        HumanInputRun()
    ]
    return tools



# def get_tool_functions():
#     tools = get_tools()
#     functions = [format_tool_to_openai_function(tl) for tl in tools]
#     return functions