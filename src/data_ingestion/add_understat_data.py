import asyncio
import aiohttp
import pandas as pd
from understat import Understat # Ensure this library is installed

async def main():
    start_year = 2018  # For the 2018-19 season
    end_year = 2024    # For the 2024-25 season

    all_player_rows = []

    # Using a try-except block for the entire async operation with the session
    try:
        async with aiohttp.ClientSession() as session:
            understat_client = Understat(session)

            for year in range(start_year, end_year + 1):
                season_formatted = f"{year}-{(year + 1) % 100:02d}"
                print(f"Fetching data for season: {season_formatted}")

                league_players = []
                try:
                    league_players = await understat_client.get_league_players(
                        league="epl",
                        season=year
                    )
                except Exception as e:
                    print(f"Could not fetch league players for season {year} ({season_formatted}): {e}")
                    continue # Skip to next season if players can't be fetched

                if not league_players:
                    print(f"No players found for season {year} ({season_formatted}). Skipping.")
                    continue

                for player_summary in league_players:
                    player_id_str = player_summary.get('id')
                    player_name = player_summary.get('player_name')

                    if not player_id_str or not player_name:
                        print(f"Skipping player with missing ID or name in season {season_formatted}: {player_summary}")
                        continue
                    
                    player_id = int(player_id_str)

                    print(f"Fetching matches for player: {player_name} (ID: {player_id}) in season {season_formatted}")

                    player_matches_data = []
                    try:
                        player_matches_data = await understat_client.get_player_matches(
                            player_id=player_id,
                            season=year
                        )
                    except Exception as e:
                        print(f"Could not fetch matches for player {player_name} (ID: {player_id}) in season {year}: {e}")
                        continue # Skip to next player if matches can't be fetched
                    
                    if not player_matches_data:
                        # This can happen if a player was listed in league_players but has no match entries (e.g. unused sub all season)
                        print(f"No match data found for player {player_name} (ID: {player_id}) in season {season_formatted}.")
                        # continue # We can choose to continue or just let it result in no rows for this player

                    for match_data in player_matches_data:
                        # Helper to safely convert to float, defaulting to 0.0 or None
                        def to_float(value, default=0.0):
                            if value is None: return default
                            try: return float(value)
                            except (ValueError, TypeError): return default

                        # Helper to safely convert to int, defaulting to 0 or None
                        def to_int(value, default=0):
                            if value is None: return default
                            try: return int(value)
                            except (ValueError, TypeError): return default

                        row_data = {
                            'goals': to_int(match_data.get('goals')),
                            'shots': to_int(match_data.get('shots')),
                            'xG': to_float(match_data.get('xG')),
                            'time': to_int(match_data.get('time')),
                            'position': match_data.get('position'),
                            'h_team': match_data.get('h_team'),
                            'a_team': match_data.get('a_team'),
                            'h_goals': to_int(match_data.get('h_goals')),
                            'a_goals': to_int(match_data.get('a_goals')),
                            'date': match_data.get('date'),
                            'id': player_id,  # Player's main ID
                            'season': year,    # Start year of the season (e.g., 2018)
                            'roster_id': to_int(match_data.get('rID')), # Player's ID in that specific match roster
                            'xA': to_float(match_data.get('xA')),
                            'assists': to_int(match_data.get('assists')),
                            'key_passes': to_int(match_data.get('key_passes')),
                            'npg': to_int(match_data.get('npg')),
                            'npxG': to_float(match_data.get('npxG')),
                            'xGChain': to_float(match_data.get('xGChain')),
                            'xGBuildup': to_float(match_data.get('xGBuildup')),
                            
                            # Additional useful ID, not explicitly in the primary list but good to have
                            'match_identifier': to_int(match_data.get('id')), # This is the match_id from Understat

                            # For index creation later
                            'season_formatted_for_idx': season_formatted, # e.g., "2018-19"
                            'player_name_for_idx': player_name,
                            'gameweek_for_idx': to_int(match_data.get('gameWeek'), default=-1) # Use -1 if gameweek is missing or invalid
                        }
                        all_player_rows.append(row_data)
                    
                    # Small delay to be polite to the server after processing each player's matches for a season
                    await asyncio.sleep(0.25) # 0.25 second delay

    except aiohttp.ClientError as e:
        print(f"A client error occurred during the HTTP session: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


    if not all_player_rows:
        print("No data collected. CSV file will not be created.")
        return

    df = pd.DataFrame(all_player_rows)

    # Create the multi-index
    # Ensure gameweek is integer type for consistent indexing if it's not -1
    df['gameweek_for_idx'] = df['gameweek_for_idx'].astype(int)
    
    try:
        df = df.set_index(['season_formatted_for_idx', 'player_name_for_idx', 'gameweek_for_idx'])
        df.index.names = ['Season', 'Player Name', 'GW']
    except KeyError as e:
        print(f"Error setting index, a required column might be missing: {e}")
        print("DataFrame columns:", df.columns)
        return


    # Define the exact order of columns for the CSV as per your request, plus the 'match_identifier'
    output_column_order = [
        'goals', 'shots', 'xG', 'time', 'position', 'h_team', 'a_team',
        'h_goals', 'a_goals', 'date', 'id', 'season', 'roster_id', 'xA',
        'assists', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup',
        'match_identifier'
    ]
    
    # Filter DataFrame to include only the specified columns
    # Check if all output columns are present in the DataFrame
    missing_cols = [col for col in output_column_order if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following requested columns are missing from the collected data and will not be in the CSV: {missing_cols}")
        # Proceed with available columns from the list
        output_column_order = [col for col in output_column_order if col in df.columns]

    if not output_column_order:
        print("No columns to save after filtering. CSV will not be created.")
        return

    df_output = df[output_column_order]

    # Save to CSV
    csv_filename = "premier_league_player_gameweek_stats_2018_2025.csv"
    try:
        df_output.to_csv(csv_filename)
        print(f"Data successfully saved to {csv_filename}")
    except Exception as e:
        print(f"Could not save DataFrame to CSV: {e}")

if __name__ == "__main__":
    # For Windows, the default event loop policy might cause issues with aiohttp.
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) # Uncomment if on Windows and facing issues
    asyncio.run(main())
