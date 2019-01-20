import peewee as pw

db = pw.SqliteDatabase(None)


class BaseModel(pw.Model):
    """
    A base model for SQLite database access.
    """
    class Meta:
        database = db


class Area(BaseModel):
    """
    A Peewee database model for a distinct Area. Area's are non-overlapping Block-wise grid sections.
    """
    block_id = pw.IntegerField()

    top_left_x = pw.FloatField()
    top_left_y = pw.FloatField()
    bot_right_x = pw.FloatField()
    bot_right_y = pw.FloatField()


class DataPeriod(BaseModel):
    """
    A Peewee database model for a distinct DataPeriod. DataPeriods are Area-related and contain datapoints across a span of time.
    """
    area = pw.ForeignKeyField(Area, related_name='dataperiods')
    date = pw.DateTimeField()

    test_ndvi_earlier = pw.BooleanField(null=True, default=0)
    test_ndvi_later = pw.BooleanField(null=True, default=0)
    test_rgb_earlier = pw.BooleanField(null=True, default=0)
    test_rgb_later = pw.BooleanField(null=True, default=0)

    area_sentinel = pw.BlobField(null=True)
    area_drone_ndvi = pw.BlobField(null=True)
    area_drone_rgb = pw.BlobField(null=True)

    block_sentinel_mean = pw.FloatField(null=True)
    block_sentinel_median = pw.FloatField(null=True)
    block_drone_ndvi_mean = pw.FloatField(null=True)
    block_drone_ndvi_median = pw.FloatField(null=True)
    block_drone_rgb_mean = pw.FloatField(null=True)
    block_drone_rgb_median = pw.FloatField(null=True)

    soil_type_cat = pw.CharField(null=True)
    soil_earthiness_cat = pw.CharField(null=True)
    soil_conductivity = pw.FloatField(null=True)
    soil_acidity = pw.FloatField(null=True)
    soil_calcium = pw.FloatField(null=True)
    soil_phosphorus = pw.FloatField(null=True)
    soil_potassium = pw.FloatField(null=True)
    soil_magnesium = pw.FloatField(null=True)
    soil_sulphur = pw.FloatField(null=True)
    soil_copper = pw.FloatField(null=True)
    soil_manganese = pw.FloatField(null=True)
    soil_zinc = pw.FloatField(null=True)
    soil_cec = pw.FloatField(null=True)
    soil_cec_ca = pw.FloatField(null=True)
    soil_cec_k = pw.FloatField(null=True)
    soil_cec_mg = pw.FloatField(null=True)
    soil_cec_na = pw.FloatField(null=True)

    weather_air_temperature_mean = pw.FloatField(null=True)
    weather_cloud_amount_mean = pw.FloatField(null=True)
    weather_dewpoint_temperature_mean = pw.FloatField(null=True)
    weather_gust_speed_mean = pw.FloatField(null=True)
    weather_horizontal_visibility_mean = pw.FloatField(null=True)
    weather_precipitation_amount_sum = pw.FloatField(null=True)
    weather_precipitation_intensity_mean = pw.FloatField(null=True)
    weather_pressure_mean = pw.FloatField(null=True)
    weather_relative_humidity_mean = pw.FloatField(null=True)
    weather_wind_direction_median = pw.FloatField(null=True)
    weather_wind_speed_mean = pw.FloatField(null=True)


class Target(BaseModel):
    """
    A Peewee database model for a distinct Target. Targets are Area-related and provide the target yield information.
    """
    area = pw.ForeignKeyField(Area, related_name='target')

    area_yield = pw.BlobField(null=True)

    block_yield_mean = pw.FloatField(null=True)
    block_yield_median = pw.FloatField(null=True)


def initialize_db(db_file_path):
    """
    Change Area, DataPeriod and Target model's database.

    Args:

        db_file_path: Path to the database file.
    """
    db.init(db_file_path)

    with db:
        try:
            db.create_tables([Area, DataPeriod, Target])
        except pw.OperationalError:
            # Tables exist
            pass
