import time
import sys
import argparse
import datetime
import numpy as np
from mmwave.dataloader import DCA1000

def save_adc_data_to_bin(adc_data, description, start_time):
    # Get current date and time
    # current_datetime = datetime.datetime.now()
    # formatted_date_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") 
    
    #Format start_time to string format
    
    # Create the filename based on current date, time, and description
    filename = f"D:\\dataset\\{start_time}_{description}.bin"

    # Save adc_data to a binary file
    adc_data.tofile(filename)
    print(f"ADC data saved to {filename}")

def main():
    # Set up the argument parser to accept num_frames and description from command line
    parser = argparse.ArgumentParser(description='Read and save ADC data from DCA1000.')
    parser.add_argument('num_frames', type=int, help='Number of frames to read from the DCA1000')
    parser.add_argument('description', type=str, help='Description for the ADC data file')

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the DCA1000
    dca = DCA1000()

    # Start timing
    start_time = datetime.datetime.now() # NEED This - Add ms-level info 

    # Read the ADC data
    adc_data = dca.read(num_frames=args.num_frames)

    # End timing
    # end_time = time.time()
    # total_time = end_time - start_time

    # Print performance statistics
    # print("Total time: ", total_time)
    # print("FPS:", int(args.num_frames) / total_time)
    # print(adc_data.shape)

    # Save the ADC data to a .bin file
    save_adc_data_to_bin(adc_data, args.description, start_time)

if __name__ == "__main__":
    main()
