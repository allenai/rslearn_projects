"""Constants for LFMC estimation."""

from enum import StrEnum

CONUS_STATES = [
    "Alabama",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]

MAX_LFMC_VALUE = 302  # 99.9th percentile of the LFMC values


class Column(StrEnum):
    """Columns in the LFMC CSV file."""

    SORTING_ID = "sorting_id"
    CONTACT = "contact"
    SITE_NAME = "site_name"
    COUNTRY = "country"
    STATE_REGION = "state_region"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    SAMPLING_DATE = "sampling_date"
    PROTOCOL = "protocol"
    LFMC_VALUE = "lfmc_value"
    SPECIES_COLLECTED = "species_collected"
    SPECIES_FUNCTIONAL_TYPE = "species_functional_type"
