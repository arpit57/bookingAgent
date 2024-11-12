from __future__ import print_function
import datetime
import os.path
import pickle
import json
from typing import Dict, Any, List, Optional
from dateutil import parser
import pytz
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI

SCOPES = ['https://www.googleapis.com/auth/calendar']
client = OpenAI()
IST = pytz.timezone('Asia/Kolkata')

class BookingAgent:
    def __init__(self):
        self.service = self.authenticate()
        self.timezone = IST
        
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

    def parse_user_intent(self, user_input: str) -> Dict[str, Any]:
        """Determine user's intention (book/update/cancel) from input"""
        prompt = f"""Analyze the user input and determine the calendar operation intent.
        Return ONLY a JSON object with no additional text. Format:
        {{
            "action": "book" | "update" | "cancel",
            "meeting_reference": "any specific meeting identifiers mentioned",
            "update_type": ["time", "participants", "duration", "title"] # only for updates
        }}
        
        User Input: {user_input}"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.choices[0].message.content.strip())

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
            return json.loads(response.choices[0].message.content.strip())
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

    def find_event(self, meeting_reference: str) -> Optional[Dict[str, Any]]:
        """Find an event based on the reference provided with improved matching"""
        try:
            # Get current time in IST
            now = datetime.datetime.now(self.timezone)
            print(f"\nCurrent time (IST): {now.strftime('%Y-%m-%d %H:%M %Z')}")
            
            # Parse the reference for date indicators
            date_prompt = f"""Extract date information from: "{meeting_reference}"
            Current time (IST) is: {now.strftime('%Y-%m-%d %H:%M')}
            Return ONLY a JSON with no additional text:
            {{
                "is_tomorrow": boolean,
                "is_today": boolean,
                "specific_date": "YYYY-MM-DD" or null,
                "specific_time": "HH:MM" or null
            }}"""
            
            date_info = json.loads(client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": date_prompt}]
            ).choices[0].message.content.strip())
            
            # Set appropriate time window based on date reference
            if date_info['is_tomorrow']:
                # Use IST midnight for tomorrow
                tomorrow_ist = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                time_min = tomorrow_ist.isoformat()
                time_max = (tomorrow_ist + datetime.timedelta(days=1)).isoformat()
            elif date_info['is_today']:
                # Use IST midnight for today
                today_ist = now.replace(hour=0, minute=0, second=0, microsecond=0)
                time_min = today_ist.isoformat()
                time_max = (today_ist + datetime.timedelta(days=1)).isoformat()
            elif date_info['specific_date']:
                # Parse specific date in IST
                specific_date = parser.parse(date_info['specific_date'])
                specific_date_ist = self.timezone.localize(specific_date.replace(hour=0, minute=0, second=0, microsecond=0))
                time_min = specific_date_ist.isoformat()
                time_max = (specific_date_ist + datetime.timedelta(days=1)).isoformat()
            else:
                # Default to searching in a 3-day window from current IST time
                time_min = now.isoformat()
                time_max = (now + datetime.timedelta(days=3)).isoformat()
            
            print(f"Searching for events between:")
            print(f"Start (IST): {parser.parse(time_min).astimezone(self.timezone).strftime('%Y-%m-%d %H:%M %Z')}")
            print(f"End (IST): {parser.parse(time_max).astimezone(self.timezone).strftime('%Y-%m-%d %H:%M %Z')}")
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                print("No events found in the specified time range")
                return None
                
            # Create a more detailed event list for matching
            event_details = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                start_dt = parser.parse(start)
                if not start_dt.tzinfo:
                    # If the datetime has no timezone, assume IST
                    start_dt = self.timezone.localize(start_dt)
                else:
                    # Convert to IST for consistent comparison
                    start_dt = start_dt.astimezone(self.timezone)
                
                event_info = {
                    'id': event['id'],
                    'summary': event.get('summary', '').lower(),
                    'description': event.get('description', '').lower(),
                    'date': start_dt.strftime('%Y-%m-%d'),
                    'time': start_dt.strftime('%H:%M'),
                    'start_datetime': start_dt.isoformat(),
                    'attendees': [att.get('email', '') for att in event.get('attendees', [])]
                }
                event_details.append(event_info)
                print(f"Found event: {event_info['summary']} at {event_info['date']} {event_info['time']} IST")
            
            # Use GPT for more contextual matching with additional context
            prompt = f"""Find the most relevant event from the list based on the reference.
            Current time (IST): {now.strftime('%Y-%m-%d %H:%M %Z')}
            Consider:
            1. Meeting title/summary matches
            2. Date matches (today/tomorrow/specific date)
            3. Time matches (if specified)
            4. If multiple events match, prefer the one closest to the mentioned time
            
            Return ONLY a JSON with the matching event ID or null if no match:
            {{"event_id": "event_id" or null, "match_reason": "brief explanation"}}
            
            Reference: {meeting_reference}
            Events: {json.dumps(event_details)}"""
            
            match_result = json.loads(client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content.strip())
            
            event_id = match_result.get('event_id')
            if not event_id:
                print(f"No matching event found. Reason: {match_result.get('match_reason', 'Unknown')}")
                return None
                
            matching_event = next((e for e in events if e['id'] == event_id), None)
            if matching_event:
                start_time = parser.parse(matching_event['start'].get('dateTime', matching_event['start'].get('date')))
                if not start_time.tzinfo:
                    start_time = self.timezone.localize(start_time)
                else:
                    start_time = start_time.astimezone(self.timezone)
                print(f"Found matching event: {matching_event.get('summary')} at {start_time.strftime('%Y-%m-%d %H:%M %Z')}")
                print(f"Match reason: {match_result.get('match_reason')}")
            return matching_event
                
        except Exception as e:
            print(f"Error finding event: {str(e)}")
            return None

    def find_available_slots(self, preferred_date: str, duration_minutes: int = 30) -> List[Dict]:
        """Find available slots on the preferred date"""
        try:
            if preferred_date is None:
                preferred_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            
            date = parser.parse(preferred_date)
            time_min = self.timezone.localize(date.replace(hour=9, minute=0))
            time_max = self.timezone.localize(date.replace(hour=17, minute=0))
            
            print(f"\nLooking for available slots on {date.strftime('%Y-%m-%d')}")
            print(f"Time range: {time_min.strftime('%H:%M')} - {time_max.strftime('%H:%M')} {str(self.timezone)}")
            
            body = {
                "timeMin": time_min.isoformat(),
                "timeMax": time_max.isoformat(),
                "timeZone": str(self.timezone),
                "items": [{"id": 'primary'}]
            }
            
            events = self.service.freebusy().query(body=body).execute()
            busy_times = events['calendars']['primary']['busy']
            
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
                current_time += datetime.timedelta(minutes=30)
            
            print(f"Found {len(free_slots)} available slots")
            return free_slots
            
        except Exception as e:
            print(f"Error finding available slots: {str(e)}")
            return []

    def create_meeting(self, meeting_details: Dict[str, Any], selected_time: str) -> Dict[str, Any]:
        """Create a meeting with proper timezone handling"""
        try:
            # Parse date and time in local timezone (IST)
            date = parser.parse(meeting_details['preferred_date'])
            time = parser.parse(selected_time).time()
            
            # Combine date and time in IST
            local_dt = datetime.datetime.combine(date, time)
            local_dt = self.timezone.localize(local_dt)
            
            event = {
                'summary': meeting_details['title'],
                'description': meeting_details['description'],
                'start': {
                    'dateTime': local_dt.isoformat(),
                    'timeZone': str(self.timezone)
                },
                'end': {
                    'dateTime': (local_dt + datetime.timedelta(minutes=meeting_details['duration_minutes'])).isoformat(),
                    'timeZone': str(self.timezone)
                }
            }
            
            if meeting_details.get('participants'):
                event['attendees'] = [{'email': email} for email in meeting_details['participants']]
            
            print(f"\nCreating meeting: {event['summary']}")
            print(f"Start time (IST): {local_dt.strftime('%Y-%m-%d %H:%M')}")
            
            created_event = self.service.events().insert(
                calendarId='primary',
                body=event,
                sendUpdates='all'
            ).execute()
            
            return {
                'status': 'success',
                'event_link': created_event.get('htmlLink'),
                'start_time': local_dt.strftime('%Y-%m-%d %H:%M'),
                'end_time': (local_dt + datetime.timedelta(minutes=meeting_details['duration_minutes'])).strftime('%Y-%m-%d %H:%M'),
                'timezone': str(self.timezone),
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

    def update_event(self, event_id: str, updated_details: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing event with timezone awareness"""
        try:
            event = self.service.events().get(calendarId='primary', eventId=event_id).execute()
            
            # Update basic details if provided
            if updated_details.get('title'):
                event['summary'] = updated_details['title']
            if updated_details.get('description'):
                event['description'] = updated_details['description']
            if updated_details.get('participants'):
                event['attendees'] = [{'email': email} for email in updated_details['participants']]
            
            # Update time if provided
            if updated_details.get('preferred_date') and updated_details.get('preferred_time'):
                date = parser.parse(updated_details['preferred_date'])
                time = parser.parse(updated_details['preferred_time']).time()
                local_dt = datetime.datetime.combine(date, time)
                local_dt = self.timezone.localize(local_dt)
                
                event['start'] = {
                    'dateTime': local_dt.isoformat(),
                    'timeZone': str(self.timezone)
                }
                event['end'] = {
                    'dateTime': (local_dt + datetime.timedelta(minutes=updated_details.get('duration_minutes', 30))).isoformat(),
                    'timeZone': str(self.timezone)
                }
            
            updated_event = self.service.events().update(
                calendarId='primary',
                eventId=event_id,
                body=event,
                sendUpdates='all'
            ).execute()
            
            return {
                'status': 'success',
                'action_taken': 'Event updated successfully',
                'event_id': event_id,
                'event_link': updated_event.get('htmlLink'),
                'timezone': str(self.timezone),
                'updated_fields': [k for k in updated_details.keys() if updated_details[k] is not None]
            }
        except Exception as e:
            return {
                'status': 'error',
                'action_taken': 'Failed to update event',
                'message': str(e)
            }

    def delete_event(self, event_id: str) -> Dict[str, Any]:
        """Delete an existing event"""
        try:
            self.service.events().delete(
                calendarId='primary',
                eventId=event_id,
                sendUpdates='all'
            ).execute()
            
            return {
                'status': 'success',
                'action_taken': 'Event cancelled successfully',
                'event_id': event_id
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'action_taken': 'Failed to cancel event'
            }

def book_appointment(user_input: str) -> Dict[str, Any]:
    """Main function to process booking requests"""
    print("\n=== Starting Calendar Operation ===")
    print(f"Processing request: {user_input}")
    
    booking_agent = BookingAgent()
    
    # First determine what the user wants to do
    intent = booking_agent.parse_user_intent(user_input)
    action = intent['action']
    
    if action == 'cancel':
        # Find and cancel the specified meeting
        event = booking_agent.find_event(intent['meeting_reference'])
        if not event:
            return {
                'status': 'error',
                'message': 'Could not find the specified meeting',
                'action_taken': 'Meeting search failed',
                'action_type': action,
                'action_details': intent
            }
            
        result = booking_agent.delete_event(event['id'])
        result.update({
            'action_type': action,
            'action_details': intent,
            'original_meeting': event.get('summary')
        })
        return result
        
    elif action == 'update':
        # Find and update the specified meeting
        event = booking_agent.find_event(intent['meeting_reference'])
        if not event:
            return {
                'status': 'error',
                'message': 'Could not find the specified meeting',
                'action_taken': 'Meeting search failed',
                'action_type': action,
                'action_details': intent
            }
            
        # Extract new details for the update
        updated_details = booking_agent.extract_meeting_details(user_input)
        
        # Filter out None values to only update specified fields
        updated_details = {k: v for k, v in updated_details.items() if v is not None}
        
        result = booking_agent.update_event(event['id'], updated_details)
        result.update({
            'action_type': action,
            'action_details': intent,
            'original_meeting': event.get('summary')
        })
        return result
        
    else:  # action == 'book'
        # Extract meeting details from user input
        meeting_details = booking_agent.extract_meeting_details(user_input)
        
        # Find available slots
        available_slots = booking_agent.find_available_slots(
            meeting_details['preferred_date'],
            meeting_details['duration_minutes']
        )
        
        if not available_slots:
            return {
                'status': 'error',
                'message': 'No available slots found for the specified date',
                'action_taken': 'Slot search failed',
                'action_type': action,
                'action_details': meeting_details
            }
        
        # Use the first available slot or preferred time if specified
        selected_time = (
            meeting_details['preferred_time'] 
            if meeting_details.get('preferred_time') 
            else available_slots[0]['start']
        )
        
        # Create the meeting
        result = booking_agent.create_meeting(meeting_details, selected_time)
        result.update({
            'action_type': action,
            'action_details': meeting_details,
            'available_slots': available_slots
        })
        return result

if __name__ == "__main__":
    # Example usage
    user_requests = [
        # "Book a team meeting tomorrow at 10 AM with arpit.singhal57@gmail.com",
        "Cancel my meeting with the team tomorrow with arpit",
        # "Update the team meeting time to 2 PM form 10 AM tomorrow"
    ]
    
    for request in user_requests:
        result = book_appointment(request)
        print(f"\nRequest: {request}")
        print(f"Result: {json.dumps(result, indent=2)}")