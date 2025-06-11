import asyncio
import aiohttp
import pandas as pd
from understat import Understat
from thefuzz import fuzz
from collections import defaultdict
import os

# --- Team Name Normalization ---
TEAM_NAME_MAPPINGS = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Cardiff": "Cardiff",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield",
    "Hull": "Hull",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Middlesbrough": "Middlesbrough",
    "Newcastle United": "Newcastle",
    "Norwich": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield Utd",
    "Southampton": "Southampton",
    "Stoke": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea": "Swansea",
    "Tottenham": "Spurs",
    "Watford": "Watford",
    "West Bromwich Albion": "West Brom",
    "West Ham": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}


def normalize_team_name(name):
    if not isinstance(name, str):
        return ""
    name_lower = name.lower().strip()
    # Apply specific mappings first
    if name_lower in TEAM_NAME_MAPPINGS:
        return TEAM_NAME_MAPPINGS[name_lower]

    # General cleaning (FC, AFC often differ)
    name_lower = name_lower.replace(" AFC", "").replace(" FC", "").replace(" & ", " and ").strip()
    # Check mappings again after general cleaning
    if name_lower in TEAM_NAME_MAPPINGS:  # In case cleaning makes it match a mapping
        return TEAM_NAME_MAPPINGS[name_lower]
    return name_lower


# Helper functions for data conversion
def to_float(value, default=0.0):
    if value is None: return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def to_int(value, default=0):
    if value is None: return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


async def get_2024_25_premier_league_player_ids(understat_client):
    """Fetches all player IDs for the 2024-25 EPL season."""
    print("Fetching player list for the 2024-25 season to use as a filter...")
    player_ids = set()
    try:
        league_players_2024 = await understat_client.get_league_players(
            league_name="epl", season=2024
        )
        for player in league_players_2024:
            player_id = player.get('id')
            if player_id:
                player_ids.add(int(player_id))
        print(f"Found {len(player_ids)} unique players in the 2024-25 Premier League season.")
    except Exception as e:
        print(f"Could not fetch 2024-25 player data: {e}. No player filtering will be applied.")
    return player_ids


async def main():
    start_year = 2020
    end_year = 2024
    all_player_rows = []
    appended_rows_count = 0

    # --- UPDATED FILE PATH ---
    # Construct the path relative to the script's location
    script_dir = os.path.dirname(__file__)
    cleaned_data_dir = os.path.abspath(os.path.join(script_dir, "../../data/raw/cleaned"))

    reference_csv_path = os.path.join(cleaned_data_dir, "cleaned_merged_seasons_team_aggregated.csv")

    # --- REVISED LOOKUP STRUCTURE ---
    # Key: (season, fpl_element_id, kickoff_date)
    # Value: List of [(opponent_name, was_home, gameweek)]
    fpl_player_game_lookup = defaultdict(list)
    fpl_player_name_to_id_lookup = {}

    try:
        gw_reference_df = pd.read_csv(reference_csv_path, low_memory=False)
        gw_reference_df['kickoff_time_date_only'] = pd.to_datetime(gw_reference_df['kickoff_time']).dt.strftime(
            '%Y-%m-%d')

        required_ref_cols = ['season_x', 'kickoff_time_date_only',
                             'opp_team_name', 'was_home',
                             'GW', 'name', 'element']

        missing_cols = [col for col in required_ref_cols if col not in gw_reference_df.columns]
        if missing_cols:
            print(
                f"CRITICAL Warning: Reference CSV ('{reference_csv_path}') is missing required columns: {missing_cols}. Cannot proceed effectively.")
            return

        # --- REVISED REFERENCE DATA PROCESSING ---
        for _, row in gw_reference_df.iterrows():
            ref_season = str(row['season_x'])
            fpl_element_id = to_int(row['element'])
            kickoff_date = str(row['kickoff_time_date_only'])
            ref_opponent_name = normalize_team_name(str(row['opp_team_name']))
            ref_gw = to_int(row['GW'])
            fpl_player_name_norm = str(row['name']).lower().strip()
            was_home_raw = row['was_home']

            if pd.isna(was_home_raw) or not ref_opponent_name:
                continue

            if isinstance(was_home_raw, str):
                ref_was_home = was_home_raw.lower() == 'true'
            else:
                ref_was_home = bool(was_home_raw)

            # Populate the game lookup using the more specific key
            fpl_player_game_lookup[(ref_season, fpl_element_id, kickoff_date)].append(
                (ref_opponent_name, ref_was_home, ref_gw)
            )

            # Populate the name-to-ID lookup
            fpl_player_name_to_id_lookup[(ref_season, fpl_player_name_norm)] = fpl_element_id

        print(f"Successfully built FPL player game and name lookups from {reference_csv_path}.")


    except FileNotFoundError:
        print(f"CRITICAL Warning: Reference file '{reference_csv_path}' not found.")
        return
    except Exception as e:
        print(f"Error processing reference file '{reference_csv_path}': {e}")
        return

    try:
        async with aiohttp.ClientSession() as session:
            understat_client = Understat(session)

            players_in_2024_season_ids = await get_2024_25_premier_league_player_ids(understat_client)
            if not players_in_2024_season_ids:
                print("Warning: Could not get 2024-25 player list. The output will not be filtered.")

            for year in range(start_year, end_year + 1):
                season_formatted = f"{year}-{(year + 1) % 100:02d}"
                print(f"\nFetching Understat data for season: {season_formatted}")

                league_players_understat = []
                try:
                    league_players_understat = await understat_client.get_league_players(
                        league_name="epl", season=year
                    )
                except Exception as e:
                    print(f"  Could not fetch Understat league players for season {year}: {e}")
                    continue

                if not league_players_understat:
                    print(f"  No Understat players found for season {year}. Skipping.")
                    continue
                print(f"  Found {len(league_players_understat)} Understat players for season {season_formatted}.")

                for player_summary in league_players_understat:
                    player_id_str = player_summary.get('id')
                    player_name_understat = player_summary.get('player_name')

                    if not player_id_str or not player_name_understat:
                        continue

                    player_id = int(player_id_str)

                    if players_in_2024_season_ids and player_id not in players_in_2024_season_ids:
                        continue

                    print(f"  Processing Understat player: {player_name_understat} (ID: {player_id_str})")

                    normalized_understat_name_for_fpl = player_name_understat.lower().strip()
                    calculated_fpl_id = -1
                    best_fuzz_score_fpl = 0

                    if (season_formatted, normalized_understat_name_for_fpl) in fpl_player_name_to_id_lookup:
                        calculated_fpl_id = fpl_player_name_to_id_lookup[
                            (season_formatted, normalized_understat_name_for_fpl)]
                        print(
                            f"[MATCHED] Exact match found: '{normalized_understat_name_for_fpl}' for season {season_formatted}.")
                    else:
                        best_fuzz_name = "N/A"
                        for (s_key, fpl_name_key), fpl_id_val in fpl_player_name_to_id_lookup.items():
                            if s_key == season_formatted:
                                score = fuzz.token_set_ratio(normalized_understat_name_for_fpl, fpl_name_key)
                                if score > best_fuzz_score_fpl:
                                    best_fuzz_score_fpl = score
                                    best_fuzz_name = fpl_name_key
                                    if score >= 70:
                                        calculated_fpl_id = fpl_id_val
                        if calculated_fpl_id != -1:
                            print(
                                f"[FUZZY MATCH] Matched '{normalized_understat_name_for_fpl}' to '{best_fuzz_name}' with score {best_fuzz_score_fpl}.")
                        else:
                            print(
                                f"[NO MATCH] Could not match '{normalized_understat_name_for_fpl}' in season {season_formatted}. Best score was {best_fuzz_score_fpl} against '{best_fuzz_name}'.")

                    all_matches_for_player_raw_understat = []
                    try:
                        all_matches_for_player_raw_understat = await understat_client.get_player_matches(
                            player_id=player_id)
                    except Exception as e:
                        continue

                    if not all_matches_for_player_raw_understat:
                        continue

                    for m_data in all_matches_for_player_raw_understat:
                        understat_match_date_full = m_data.get('date')
                        if not understat_match_date_full: continue

                        try:
                            understat_match_date_only = pd.to_datetime(understat_match_date_full).strftime('%Y-%m-%d')
                        except Exception:
                            if isinstance(understat_match_date_full, str) and " " in understat_match_date_full:
                                understat_match_date_only = understat_match_date_full.split(" ")[0]
                            else:
                                continue

                        # --- REVISED MATCH VALIDATION LOGIC ---
                        validated_gw = -1
                        match_validated_as_epl = False

                        # Only proceed if we have a valid FPL ID for the player
                        if calculated_fpl_id != -1:
                            lookup_key = (season_formatted, calculated_fpl_id, understat_match_date_only)
                            possible_fpl_games = fpl_player_game_lookup.get(lookup_key)

                            if possible_fpl_games:
                                understat_h_team_raw = m_data.get('h_team')
                                understat_a_team_raw = m_data.get('a_team')
                                if not understat_h_team_raw or not understat_a_team_raw:
                                    continue

                                norm_understat_h = normalize_team_name(understat_h_team_raw)
                                norm_understat_a = normalize_team_name(understat_a_team_raw)

                                # Check against the specific FPL games for that player on that day
                                for ref_opponent_name, ref_was_home, ref_gw in possible_fpl_games:
                                    understat_opponent_name = norm_understat_a if ref_was_home else norm_understat_h
                                    opponent_match_score = fuzz.token_set_ratio(understat_opponent_name, ref_opponent_name)

                                    if opponent_match_score >= 90:
                                        validated_gw = ref_gw
                                        match_validated_as_epl = True
                                        break  # Match found, no need to check further

                        if match_validated_as_epl:
                            row_data = {
                                'goals': to_int(m_data.get('goals')), 'shots': to_int(m_data.get('shots')),
                                'xG': to_float(m_data.get('xG')), 'time': to_int(m_data.get('time')),
                                'position': m_data.get('position'), 'h_team': understat_h_team_raw,
                                'a_team': understat_a_team_raw, 'h_goals': to_int(m_data.get('h_goals')),
                                'a_goals': to_int(m_data.get('a_goals')), 'date': understat_match_date_full,
                                'id': player_id, 'season': year,
                                'roster_id': to_int(m_data.get('rID')), 'xA': to_float(m_data.get('xA')),
                                'assists': to_int(m_data.get('assists')),
                                'key_passes': to_int(m_data.get('key_passes')),
                                'npg': to_int(m_data.get('npg')), 'npxG': to_float(m_data.get('npxG')),
                                'xGChain': to_float(m_data.get('xGChain')),
                                'xGBuildup': to_float(m_data.get('xGBuildup')),
                                'match_identifier': to_int(m_data.get('id')),
                                'fpl_id': to_int(calculated_fpl_id, default=-1),
                                'season_formatted_for_idx': season_formatted,
                                'player_name_for_idx': player_name_understat,
                                'gameweek_for_idx': to_int(validated_gw, default=-1)
                            }
                            all_player_rows.append(row_data)
                            appended_rows_count += 1

                    if len(all_matches_for_player_raw_understat) > 0:
                        await asyncio.sleep(0.02)

    except aiohttp.ClientError as e:
        print(f"A client error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f"\n--- Processing Complete. Total rows appended: {appended_rows_count} ---")

    if not all_player_rows:
        print("No data collected. CSV file will not be created.")
        return

    df = pd.DataFrame(all_player_rows)
    if df.empty:
        print("DataFrame is empty after collecting rows. CSV file will not be created.")
        return

    df = df.sort_values(by=['season_formatted_for_idx', 'player_name_for_idx', 'gameweek_for_idx'])

    try:
        df = df.set_index(['season_formatted_for_idx', 'player_name_for_idx', 'gameweek_for_idx'])
        df.index.names = ['Season', 'Player Name', 'GW']
    except Exception as e:
        print(f"Error setting index: {e}")
        print("DataFrame columns:", df.columns)
        print("Saving DataFrame without index due to error.")
        try:
            # --- UPDATED FILE PATH ---
            no_index_path = os.path.join(cleaned_data_dir, "premier_league_player_gameweek_stats_NO_INDEX.csv")
            df.to_csv(no_index_path, index=False)
        except Exception as e_noindex:
            print(f"Could not save DataFrame without index: {e_noindex}")
        return

    output_column_order = [
        'goals', 'shots', 'xG', 'time', 'position', 'h_team', 'a_team',
        'h_goals', 'a_goals', 'date', 'id', 'season', 'fpl_id', 'roster_id', 'xA',
        'assists', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup',
        'match_identifier'
    ]

    final_output_columns = [col for col in output_column_order if col in df.columns]
    missing_cols_output = [col for col in output_column_order if col not in df.columns]
    if missing_cols_output:
        print(f"Warning: Columns missing from final output: {missing_cols_output}")

    if not final_output_columns:
        print("No columns for final output. CSV not created.")
        return

    df_output = df[final_output_columns]

    # --- UPDATED FILE PATH ---
    csv_filename = os.path.join(cleaned_data_dir,
                                "understat_2020_24.csv")

    try:
        # Ensure the directory exists before saving
        os.makedirs(cleaned_data_dir, exist_ok=True)
        df_output.to_csv(csv_filename)
        print(f"Data successfully saved to {csv_filename}")
    except Exception as e:
        print(f"Could not save DataFrame to CSV: {e}")


if __name__ == "__main__":
    asyncio.run(main())