from __future__ import print_function
import datetime
import os.path
import pickle
import json
from typing import Dict, Any, List, Optional
from dateutil import parser
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI

SCOPES = ['https://www.googleapis.com/auth/calendar']
client = OpenAI()

class BookingAgent:
    def __init__(self):
        self.service = self.authenticate()
        
    def authenticate(self):
        """Authenticate with Google Calendar"""
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
                
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
                
        return build('calendar', 'v3', credentials=creds)

    def extract_meeting_details(self, user_input: str) -> Dict[str, Any]:
        """Extract meeting details from user input using GPT-4"""
        prompt = f"""Extract meeting details from the following user input. 
        Return ONLY a JSON object with no additional text. Format:
        {{
            "title": "meeting title",
            "duration_minutes": meeting duration in minutes (default to 30 if not specified),
            "preferred_date": "YYYY-MM-DD format, use tomorrow's date if 'tomorrow' is mentioned",
            "preferred_time": "HH:MM format in 24h",
            "description": "meeting description",
            "participants": ["email1", "email2"]
        }}
        
        Today's date is {datetime.datetime.now().strftime('%Y-%m-%d')}.
        If any information is missing, use null for that field.
        
        User Input: {user_input}"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            response_text = response.choices[0].message.content.strip()
            print(f"Extracted details: {response_text}")
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Error parsing meeting details: {e}")
            tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
            return {
                "title": "Meeting",
                "duration_minutes": 30,
                "preferred_date": tomorrow.strftime('%Y-%m-%d'),
                "preferred_time": "09:00",
                "description": user_input,
                "participants": []
            }

    def find_available_slots(self, preferred_date: str, duration_minutes: int = 30) -> List[Dict]:
        """Find available slots on the preferred date"""
        try:
            # Parse the preferred date
            if preferred_date is None:
                preferred_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            
            date = parser.parse(preferred_date)
            time_min = date.replace(hour=9, minute=0)  # Start at 9 AM
            time_max = date.replace(hour=17, minute=0)  # End at 5 PM
            
            print(f"\nLooking for available slots on {date.strftime('%Y-%m-%d')}")
            print(f"Time range: {time_min.strftime('%H:%M')} - {time_max.strftime('%H:%M')}")
            
            body = {
                "timeMin": time_min.isoformat() + 'Z',
                "timeMax": time_max.isoformat() + 'Z',
                "timeZone": 'UTC',
                "items": [{"id": 'primary'}]
            }
            
            events = self.service.freebusy().query(body=body).execute()
            busy_times = events['calendars']['primary']['busy']
            
            print(f"Found {len(busy_times)} busy slots")
            
            # Find free slots
            free_slots = []
            current_time = time_min
            while current_time + datetime.timedelta(minutes=duration_minutes) <= time_max:
                slot_start = current_time
                slot_end = current_time + datetime.timedelta(minutes=duration_minutes)
                slot_busy = False
                
                for busy in busy_times:
                    busy_start = parser.isoparse(busy['start'])
                    busy_end = parser.isoparse(busy['end'])
                    if (slot_start < busy_end) and (slot_end > busy_start):
                        slot_busy = True
                        break
                        
                if not slot_busy:
                    free_slots.append({
                        'start': slot_start.strftime('%H:%M'),
                        'end': slot_end.strftime('%H:%M')
                    })
                current_time += datetime.timedelta(minutes=30)  # 30-minute intervals
            
            print(f"Found {len(free_slots)} available slots")
            for slot in free_slots:
                print(f"Available: {slot['start']} - {slot['end']}")
            
            return free_slots
            
        except Exception as e:
            print(f"Error finding available slots: {str(e)}")
            return []

    def create_meeting(self, meeting_details: Dict[str, Any], selected_time: str) -> Dict[str, Any]:
        """Create a meeting at the selected time"""
        try:
            date = parser.parse(meeting_details['preferred_date'])
            time = parser.parse(selected_time).time()
            start_time = datetime.datetime.combine(date, time)
            
            event = {
                'summary': meeting_details['title'],
                'description': meeting_details['description'],
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': (start_time + datetime.timedelta(minutes=meeting_details['duration_minutes'])).isoformat(),
                    'timeZone': 'UTC',
                }
            }
            
            if meeting_details.get('participants'):
                event['attendees'] = [{'email': email} for email in meeting_details['participants']]
            
            print(f"\nCreating meeting: {event['summary']}")
            print(f"Date: {start_time.strftime('%Y-%m-%d')}")
            print(f"Time: {start_time.strftime('%H:%M')} - {(start_time + datetime.timedelta(minutes=meeting_details['duration_minutes'])).strftime('%H:%M')}")
            
            created_event = self.service.events().insert(
                calendarId='primary',
                body=event,
                sendUpdates='all'
            ).execute()
            
            return {
                'status': 'success',
                'event_link': created_event.get('htmlLink'),
                'start_time': start_time.strftime('%Y-%m-%d %H:%M'),
                'end_time': (start_time + datetime.timedelta(minutes=meeting_details['duration_minutes'])).strftime('%Y-%m-%d %H:%M'),
                'action_taken': 'Meeting created successfully',
                'meeting_title': event['summary'],
                'participants': meeting_details.get('participants', [])
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'action_taken': 'Failed to create meeting'
            }

def book_appointment(user_input: str) -> Dict[str, Any]:
    """Main function to process booking requests"""
    print("\n=== Starting Booking Process ===")
    print(f"Processing request: {user_input}")
    
    booking_agent = BookingAgent()
    
    # Extract meeting details
    meeting_details = booking_agent.extract_meeting_details(user_input)
    
    # Find available slots
    available_slots = booking_agent.find_available_slots(
        meeting_details['preferred_date'],
        meeting_details['duration_minutes']
    )
    
    if not available_slots:
        return {
            'status': 'error',
            'message': 'No available slots found for the requested date.',
            'action_taken': 'Slot search failed',
            'requested_date': meeting_details['preferred_date']
        }
    
    # Use GPT to select the best slot based on user's preferred time
    slot_selection_prompt = f"""Given the following available time slots and user's preferred time, 
    return ONLY the start time of the best matching slot in HH:MM format with no additional text.
    
    User preferred time: {meeting_details['preferred_time']}
    Available slots: {json.dumps(available_slots)}"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": slot_selection_prompt}]
    )
    
    selected_time = response.choices[0].message.content.strip()
    print(f"\nSelected time slot: {selected_time}")
    
    # Book the meeting
    result = booking_agent.create_meeting(meeting_details, selected_time)
    
    # Add context for the main workflow
    result['meeting_details'] = meeting_details
    result['available_slots'] = available_slots
    result['selected_time'] = selected_time
    
    print("\n=== Booking Process Complete ===")
    print(f"Final status: {result['status']}")
    print(f"Action taken: {result.get('action_taken', 'No action recorded')}")
    
    return result

# For testing
if __name__ == '__main__':
    test_input = "I want to schedule a team meeting for tomorrow at 4pm for 1 hour. The topic is weekly planning with arpit.singhal57@gmail.com"
    result = book_appointment(test_input)
    print("\nFinal Result:")
    print(json.dumps(result, indent=2))