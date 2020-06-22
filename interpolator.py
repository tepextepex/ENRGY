import numpy as np
from datetime import datetime


input_arrays = {
	"20190727": np.arange(16).reshape(4, 4),
	"20190803": np.arange(4, 20).reshape(4, 4)
}

needed_date = "20190731"
# needed_date = "20190710"


def interpolate_array(arrays_dict, needed_date):
	needed_date = datetime.strptime(needed_date, "%Y%m%d")
	before, after = _get_closest_dates(arrays_dict, needed_date)

	ar0 = input_arrays[before.strftime("%Y%m%d")]
	ar1 = input_arrays[after.strftime("%Y%m%d")]

	result = ar0 + (needed_date - before).days * (ar1 - ar0) / (after - before).days  # just a linear interpolation

	return result


def _get_closest_dates(arrays_dict, needed_date):
	if not isinstance(needed_date, datetime):
		needed_date = datetime.strptime(needed_date, "%Y%m%d")

	dates = [datetime.strptime(date_str, "%Y%m%d") for date_str in [*arrays_dict]]  # [*arrays_dict] returns a list containing keys from input dict

	dates_before = [date for date in dates if date < needed_date]
	dates_after = [date for date in dates if date > needed_date]

	if len(dates_before) == 0 or len(dates_after) == 0:
		raise ValueError("Passed date is outside of the possible interpolation range!")

	closest_date_before = max(dates_before)
	closest_date_after = min(dates_after)

	return closest_date_before, closest_date_after


if __name__ == "__main__":
	# print(_get_closest_dates(input_arrays, needed_date))
	interpolate_array(input_arrays, needed_date)
