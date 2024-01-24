#!/bin/bash

docker compose run step_to_obj_converter
docker compose run obj_to_usd_converter
docker compose run config_generator_for_dataset_generation
