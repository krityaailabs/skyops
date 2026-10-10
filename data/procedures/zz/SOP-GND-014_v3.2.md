---
doc_id: SOP-GND-014
title: Delay care: meals, hotels and rebooking
version: 3.2
owner: Ground Operations
tenant: zz
roles: agent, duty_manager, ground_staff
classification: internal
effective: 2026-10-01
withdrawn:
supersedes: SOP-GND-009
---
# Delay care: meals, hotels and rebooking

## Purpose

This procedure tells hub staff what care ZZ passengers receive when a departure from HUB is delayed or cancelled. It replaces SOP-GND-009 (hotel vouchers for overnight delays), which is withdrawn. Care is offered by the gate or transfer desk agent; approvals above the limits in this procedure are given by the duty manager on shift.

## When care applies

Care depends on the delay at the gate, measured from the scheduled departure time to the latest estimated departure time.

| Delay at the gate | Care offered | Who authorises |
|---|---|---|
| 2 hours or more | Meal voucher, 15 EUR per passenger | Gate agent |
| 3 hours or more | Second meal voucher, 15 EUR per passenger | Gate agent |
| Departure after 22:00 local, or overnight | Hotel room and transport | Gate agent up to 150 EUR, duty manager above |
| Cancellation | Rebooking, meal vouchers, hotel if overnight | Transfer desk |

## Meal vouchers

Issue meal vouchers with the issue_voucher tool, kind "meal", one per booking and passenger. Vouchers are valid at airside outlets until the new departure time. Do not issue cash instead of a voucher.

## Hotel accommodation

Follow these steps for a hotel room when the departure moves past 22:00 local or the flight is cancelled overnight.

1. Confirm the latest estimated departure with get_flight_status, and check that the delay is recorded with a reason.
2. Check room availability at the contracted hotels in the ground operations portal.
3. Book the room for the passenger's party, one room per booking unless the party is larger than three.
4. Issue the hotel voucher with the issue_voucher tool, kind "hotel".
5. A hotel voucher above 150 EUR needs the duty manager's approval before it is issued. The tool returns approval_required above this limit; do not split a stay into two vouchers to stay under it.
6. Arrange transport to the hotel and tell the passenger the pick-up point and the time to be back at the airport.

## Rebooking

Rebook passengers on the next ZZ departure with seats in the same cabin. If none departs within 12 hours, offer partner flights through the transfer desk. Passengers who no longer wish to travel are offered a refund of the unused ticket.

## Records

Record every voucher and hotel booking against the disruption reference in the ground operations portal before the end of the shift.
