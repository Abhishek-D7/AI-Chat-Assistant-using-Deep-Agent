from langchain.tools import tool
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import asyncio
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
import uuid
import logging
import re
import pytz
from app.config import Config

logger = logging.getLogger(__name__)

class GoogleCalendarManager:
    """Google Calendar API Manager - Async Wrapper"""
    
    def __init__(self):
        """Initialize Google Calendar manager"""
        self.credentials_file = Config.GOOGLE_CREDENTIALS_FILE
        self.token_file = Config.GOOGLE_TOKEN_FILE
        self.scopes = ['https://www.googleapis.com/auth/calendar']
        self.service = None
        
        # Calendar configuration from Config
        self.calendar_id = "primary"
        self.meeting_duration = Config.MEETING_DURATION_MINUTES
        self.buffer_time = Config.BUFFER_TIME_MINUTES
        self.working_hours = Config.get_working_hours()
        self.timezone = pytz.timezone(Config.DEFAULT_TIMEZONE)
        
        # Authenticate synchronously on init (or could be lazy loaded)
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Calendar with Robust Error Handling"""
        creds = None
        
        try:
            # 1. Try to load existing token
            if os.path.exists(self.token_file):
                logger.info("📝 Loading Google Calendar token...")
                try:
                    creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
                except Exception as e:
                    logger.warning(f"⚠️ Corrupt token file: {e}")
                    creds = None
            
            # 2. Check validity / Refresh
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        logger.info("🔄 Refreshing token...")
                        creds.refresh(Request())
                    except (RefreshError, Exception) as e:
                        logger.warning(f"⚠️ Token refresh failed: {e}")
                        logger.info("🗑️ Deleting invalid token.json to force re-login...")
                        if os.path.exists(self.token_file):
                            os.remove(self.token_file)
                        creds = None
                
                # 3. New Login (if no creds or refresh failed)
                if not creds:
                    logger.info("🔐 Requesting new authorization (Browser will open)...")
                    if not os.path.exists(self.credentials_file):
                        logger.error(f"❌ Missing {self.credentials_file}. Cannot authenticate.")
                        # Don't raise here to allow app to start without calendar
                        return
                        
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, self.scopes
                    )
                    creds = flow.run_local_server(port=0)
                
                # 4. Save valid token
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())
            
            self.service = build('calendar', 'v3', credentials=creds)
            logger.info("✅ Google Calendar authenticated")
        
        except Exception as e:
            logger.error(f"❌ Calendar auth failed completely: {e}")
            self.service = None

    async def is_slot_available(self, date_str: str, time_slot: str) -> bool:
        """Check if a specific time slot is available (Non-blocking)"""
        if not self.service: return False
        
        return await asyncio.to_thread(self._is_slot_available_sync, date_str, time_slot)

    def _is_slot_available_sync(self, date_str: str, time_slot: str) -> bool:
        """Synchronous implementation of availability check"""
        try:
            # Parse datetime
            if date_str.lower() == "tomorrow":
                target_date = datetime.now() + timedelta(days=1)
                date_str = target_date.strftime("%Y-%m-%d")
            
            # Parse time
            time_obj = datetime.strptime(time_slot.split(" - ")[0].strip(), "%I:%M %p").time()
            meeting_start = datetime.combine(
                datetime.strptime(date_str, "%Y-%m-%d").date(),
                time_obj
            )
            meeting_end = meeting_start + timedelta(minutes=self.meeting_duration)
            
            # Set timezone
            meeting_start = self.timezone.localize(meeting_start)
            meeting_end = self.timezone.localize(meeting_end)
            
            logger.info(f"🔍 Checking availability: {meeting_start} to {meeting_end}")
            
            # Query calendar
            events = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=meeting_start.isoformat(),
                timeMax=meeting_end.isoformat(),
                singleEvents=True
            ).execute().get('items', [])
            
            is_available = len(events) == 0
            logger.info(f"{'✅' if is_available else '❌'} Slot is {'available' if is_available else 'booked'}")
            
            return is_available
        
        except Exception as e:
            logger.error(f"❌ Error checking availability: {e}")
            return False
    
    async def get_available_slots(self, date_str: str = None, num_slots: int = 5) -> List[str]:
        """Get available slots from Google Calendar (Non-blocking)"""
        if not self.service: return []
        
        return await asyncio.to_thread(self._get_available_slots_sync, date_str, num_slots)

    def _get_available_slots_sync(self, date_str: str = None, num_slots: int = 5) -> List[str]:
        """Synchronous implementation of getting slots"""
        try:
            if not date_str or date_str.lower() == "tomorrow":
                date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            logger.info(f"📅 Fetching slots for {date_str}")
            
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            start_time = target_date.replace(hour=self.working_hours["start"], minute=0, second=0)
            end_time = target_date.replace(hour=self.working_hours["end"], minute=0, second=0)
            
            start_time = self.timezone.localize(start_time)
            end_time = self.timezone.localize(end_time)
            
            # Get events
            events = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=start_time.isoformat(),
                timeMax=end_time.isoformat(),
                singleEvents=True,
                orderBy='startTime'
            ).execute().get('items', [])
            
            booked_times = []
            for event in events:
                if 'start' in event and 'end' in event:
                    start = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(event['end']['dateTime'].replace('Z', '+00:00'))
                    booked_times.append((start, end))
            
            available_slots = []
            current_time = start_time
            
            while len(available_slots) < num_slots and current_time < end_time:
                slot_end = current_time + timedelta(minutes=self.meeting_duration)
                
                is_available = True
                for booked_start, booked_end in booked_times:
                    if current_time < booked_end and slot_end > booked_start:
                        is_available = False
                        current_time = booked_end + timedelta(minutes=self.buffer_time)
                        break
                
                if is_available and slot_end <= end_time:
                    slot_str = current_time.strftime("%I:%M %p") + " - " + slot_end.strftime("%I:%M %p")
                    available_slots.append(slot_str)
                
                current_time += timedelta(minutes=self.meeting_duration + self.buffer_time)
            
            return available_slots if available_slots else ["9:00 AM - 10:00 AM", "10:00 AM - 11:00 AM"]
        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return []
    
    async def book_meeting(self, user_email: str, slot: str, meeting_title: str = "B2B Consultation", date_str: str = None) -> Dict:
        """Book a meeting (Non-blocking)"""
        if not self.service: raise Exception("Calendar service not authenticated")

        return await asyncio.to_thread(self._book_meeting_sync, user_email, slot, meeting_title, date_str)

    def _book_meeting_sync(self, user_email: str, slot: str, meeting_title: str, date_str: str) -> Dict:
        """Synchronous implementation of booking"""
        try:
            if not date_str or date_str.lower() == "tomorrow":
                date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            logger.info(f"📅 Booking: {meeting_title} at {slot} on {date_str}")
            
            # Parse time
            time_parts = slot.split(" - ")
            start_time_str = time_parts[0].strip()
            
            meeting_start = datetime.strptime(f"{date_str} {start_time_str}", "%Y-%m-%d %I:%M %p")
            meeting_end = meeting_start + timedelta(minutes=self.meeting_duration)
            
            meeting_start = self.timezone.localize(meeting_start)
            meeting_end = self.timezone.localize(meeting_end)
            
            # Create event
            event = {
                'summary': meeting_title,
                'start': {'dateTime': meeting_start.isoformat(), 'timeZone': 'UTC'},
                'end': {'dateTime': meeting_end.isoformat(), 'timeZone': 'UTC'},
                'conferenceData': {
                    'createRequest': {
                        'requestId': str(uuid.uuid4())
                    }
                },
                'attendees': [{'email': user_email}],
            }
            
            event = self.service.events().insert(
                calendarId=self.calendar_id,
                body=event,
                conferenceDataVersion=1,
                sendUpdates='all'
            ).execute()
            
            meet_link = "https://meet.google.com"
            if 'conferenceData' in event and 'entryPoints' in event['conferenceData']:
                for entry in event['conferenceData']['entryPoints']:
                    if entry['entryPointType'] == 'video':
                        meet_link = entry['uri']
            
            logger.info(f"✅ Meeting booked: {event['id']}")
            
            return {
                "booking_id": event['id'],
                "user_email": user_email,
                "title": meeting_title,
                "slot": slot,
                "date": date_str,
                "meet_link": meet_link,
                "status": "confirmed"
            }
        
        except Exception as e:
            logger.error(f"❌ Booking failed: {e}")
            raise


    async def cancel_meeting(self, user_email: str, reason: str) -> Optional[str]:
        """Cancel a meeting based on user email and reason (Non-blocking)"""
        if not self.service: return None
        
        return await asyncio.to_thread(self._cancel_meeting_sync, user_email, reason)

    def _cancel_meeting_sync(self, user_email: str, reason: str) -> Optional[str]:
        """Synchronous implementation of meeting cancellation"""
        try:
            logger.info(f"🗑️ Attempting to cancel meeting for {user_email} with reason: {reason}")
            
            # Search for future events
            now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
            
            events_result = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=now,
                maxResults=20,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])

            for event in events:
                # Check attendees
                attendees = event.get('attendees', [])
                attendee_emails = [a.get('email') for a in attendees]
                
                # Check if user is an attendee and reason matches summary
                # We use a loose match for reason in summary
                if user_email in attendee_emails and reason.lower() in event.get('summary', '').lower():
                    logger.info(f"✅ Found meeting to cancel: {event.get('summary')} at {event.get('start')}")
                    
                    self.service.events().delete(
                        calendarId=self.calendar_id,
                        eventId=event['id'],
                        sendUpdates='all'
                    ).execute()
                    
                    return f"{event.get('summary')} on {event.get('start').get('dateTime') or event.get('start').get('date')}"

            logger.info("⚠️ No matching meeting found to cancel.")
            return None

        except Exception as e:
            logger.error(f"❌ Error cancelling meeting: {e}")
            return None


# Global instance
try:
    calendar_manager = GoogleCalendarManager()
except Exception as e:
    logger.error(f"⚠️ Failed to initialize Calendar Manager: {e}")
    calendar_manager = None


@tool
async def booking_agent_tool(
    date: str, 
    time: str, 
    email: str,
    name: str,
    contact: str,
    company_name: str,
    reason: str = "General Consultation", 
    reschedule: bool = False
) -> str:
    """
    Booking agent tool. Use this to book appointments.
    
    Args:
        date: The date for the appointment (e.g., "2025-11-27", "tomorrow", or "next Monday")
        time: The time for the appointment (MUST be between 9:00 AM and 5:00 PM)
        email: The user's email address.
        name: The user's full name.
        contact: The user's phone number or contact details.
        company_name: The user's company name.
        reason: The reason or topic for the appointment.
        reschedule: Set to True to reschedule an existing appointment (cancels previous one).
    """

    logger.info(f"📥 Booking Request: {name} ({company_name}) - {date} @ {time}. Reason: {reason}")

    # 1. Validate Time (9AM - 5PM)
    try:
        # Simple parse to check hour
        # Formats: 9:00 AM, 09:00, 14:00
        t_str = time.upper().replace(".","")
        if "AM" in t_str or "PM" in t_str:
             t_obj = datetime.strptime(t_str, "%I:%M %p")
        else:
             t_obj = datetime.strptime(t_str, "%H:%M")
        
        # Check range 9-17
        if t_obj.hour < 9 or t_obj.hour >= 17:
             return f"⚠️ Time {time} is out of business hours (9 AM - 5 PM). Please choose a valid time."
    except Exception as e:
        logger.warning(f"Time validation warning: {e}")
        # Proceed if parse fails, let calendar manager handle it or fail there


    # Normalize date if needed
    date_str = date
    if date_str and isinstance(date_str, str):
        date_str = date_str.lower().strip()
        try:
            today = datetime.now()
            target_date = None
            
            if date_str == "today":
                target_date = today
            elif date_str == "tomorrow":
                target_date = today + timedelta(days=1)
            elif date_str.startswith("next ") or date_str in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                target_day = date_str.replace("next ", "").strip()
                
                if target_day in weekdays:
                    current_weekday = today.weekday()
                    target_weekday = weekdays.index(target_day)
                    
                    days_ahead = target_weekday - current_weekday
                    if days_ahead <= 0: # Target day already happened this week
                        days_ahead += 7
                    if "next " in date_str: 
                         # Logic: If today is Monday and user says "next Monday", usually means 7 days later.
                         if days_ahead < 7:
                             days_ahead += 7
                    
                    target_date = today + timedelta(days=days_ahead)
            
            if target_date:
                # Format to YYYY-MM-DD
                date_str = target_date.strftime("%Y-%m-%d")
            else:
                # Try simple parse
                for fmt in ["%d %B %Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"]:
                    try:
                        parsed = datetime.strptime(date, fmt)
                        date_str = parsed.strftime("%Y-%m-%d")
                        break
                    except:
                        continue
        except Exception as e:
            logger.warning(f"⚠️ Date parsing warning: {e}")
    
    time_str = time

    if not date_str or not time_str:
        return "I need both date and time to book your appointment."
    
    if not calendar_manager:
        return "Calendar system is currently offline."

    try:
        # Handle Rescheduling (Cancel old meeting first)
        cancel_msg = ""
        if reschedule:
            cancelled_meeting = await calendar_manager.cancel_meeting(email, reason)
            if cancelled_meeting:
                cancel_msg = f"🗑️ **Previous meeting cancelled:** {cancelled_meeting}\n\n"
            else:
                cancel_msg = "⚠️ Could not find a previous meeting to cancel, but proceeding with new booking.\n\n"

        # Check availability (ASYNC)
        is_available = await calendar_manager.is_slot_available(date_str, time_str)

        if not is_available:
            # Get slots (ASYNC)
            available_slots = await calendar_manager.get_available_slots(date_str)
            return (
                f"{cancel_msg}"
                f"⛔ The slot **{time_str} on {date_str}** is not available.\n"
                f"Available slots:\n" +
                "\n".join(f"- {slot}" for slot in available_slots)
            )

        # Book appointment (ASYNC)
        # Calculate End Time
        # Note: simplistic parsing, production should correspond to strptime used above
        try:
             t_obj = datetime.strptime(time_str.upper().replace(".",""), "%I:%M %p")
        except:
             # Fallback
             return "Invalid time format. Please use 'HH:MM AM/PM'."

        slot_end = (t_obj + timedelta(minutes=Config.MEETING_DURATION_MINUTES)).strftime("%I:%M %p")
        
        # Enhanced Title with Company & Name
        full_title = f"{company_name}: {reason} ({name})"

        booking = await calendar_manager.book_meeting(
            user_email=email,
            date_str=date_str,
            slot=f"{time_str} - {slot_end}",
            meeting_title=full_title
        )

        return (
            f"{cancel_msg}"
            f"✅ Appointment booked!\n\n"
            f"📌 **Topic:** {reason}\n"
            f"🏢 **Company:** {company_name}\n"
            f"👤 **Name:** {name}\n"
            f"📅 **Date:** {booking['date']}\n"
            f"⏰ **Time:** {booking['slot']}\n"
            f"🔗 **Meet Link:** {booking['meet_link']}"
        )

    except Exception as e:
        logger.error(f"❌ Error in booking_agent_tool: {e}", exc_info=True)
        return "Something went wrong while booking. Try again."
